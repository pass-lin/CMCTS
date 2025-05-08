# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:45:07 2024

@author: Administrator
"""

import os
from transformers import AutoTokenizer
import numpy as np
import gc
from vllm import LLM, SamplingParams
import difflib
import re
from copy import deepcopy
from wrapt_timeout_decorator import timeout


@timeout(1)
def execute(a, b):
    exec(a, b)


def find_closest(str1, str_list):
    # 创建SequenceMatcher对象
    seq_match = difflib.SequenceMatcher()
    seq_match.set_seq1(str1)

    # 初始化最接近的字符串和最大匹配度
    closest_str = None
    max_ratio = 0

    # 遍历字符串列表，找到匹配度最高的字符串
    for s in str_list:
        seq_match.set_seq2(s)
        ratio = seq_match.ratio()
        if ratio > max_ratio:
            max_ratio = ratio
            closest_str = s

    return closest_str, max_ratio


def get_deberta_retrieval_model(
    logits_config_path, logits_weights_path, logits_dict_path
):
    """
    创建并返回一个DeBERTa模型用于检索任务，以及相应的tokenizer。

    参数:
    - logits_config_path (str): DeBERTa模型配置文件的路径。
    - logits_weights_path (str): DeBERTa模型权重文件的路径。
    - logits_dict_path (str): 词汇表文件的路径，用于初始化tokenizer。

    返回:
    - encoder (keras.Model): 用于检索任务的DeBERTa模型。
    - tokenizer (SpTokenizer): 用于文本token化的tokenizer。
    """

    import keras
    from bert4keras3.tokenizers import SpTokenizer
    from bert4keras3.layers import GlobalAveragePooling1D
    from bert4keras3.models import build_transformer_model

    dtype = keras.config.dtype_policy()
    keras.config.set_dtype_policy("float32")
    tokenizer = SpTokenizer(logits_dict_path)

    deberta = build_transformer_model(
        config_path=logits_config_path,
        keras_weights_path=logits_weights_path,
        model="deberta",
        return_keras_model=True,
        dropout_rate=0.3,
        with_mlm=False,
    )
    mask = deberta.get_layer("Padding-Mask").output

    z1 = GlobalAveragePooling1D(name="Pooling-Last")(deberta.output[0], mask=mask[:, 0])
    z2 = GlobalAveragePooling1D(name="Pooling-First")(
        deberta.get_layer("Transformer-0-FeedForward-Norm").output, mask=mask[:, 0]
    )
    encoder = keras.Model(deberta.inputs, (z1 + z2) / 2)
    keras.config.set_dtype_policy(dtype)
    encoder.compile(jit_compile=True)
    return encoder, tokenizer


max_output_str = 256


def eval_trajectory(trajectory):
    question = trajectory.split("\n")[0].split(":")[-1].split(",")[-1]
    states = trajectory.split("\n")
    if len(states) % 2 == 0:
        return False
    while states[-1] == "":
        states.pop(-1)
    for j in range(1, len(states) - 4, 2):
        distance = difflib.SequenceMatcher(
            None, states[j].split(":")[1], question
        ).ratio()
        if distance > 0.8:
            return True
    return False


def eval_quality_rule(trajectory, config):
    trajectory_lits = trajectory.split("\n")
    if len(trajectory[:-1].split("\n")) % 2 == 0:
        return False

    for t in trajectory_lits[1:-1]:
        if config.answer_prefix + " {idx}." in t:
            if t.lower().count(config.answer_prefix.lower()) == 1:
                return False
    return not eval_trajectory(trajectory)


class GenerateModel:
    def __init__(
        self,
        model_name: str,
        temperature=1,
        top_p: float = 0.7,
        max_tokens: int = 256,
        top_k: int = 50,
        stop: list = ["\n"],
        stop_token_ids=None,
        select_tokens=[" Yes", " No"],
        gpu_memory_utilization=0.9,
        enable_lora=False,
        reward_model=None,
        max_generate_model_len=None,
        max_model_len=None,
        use_tqdm=True,
    ):
        self.use_tqdm = use_tqdm
        if stop_token_ids is None:
            if "qwen" in model_name.lower():
                stop_token_ids = [151645, 151643]
            elif "yi" in model_name.lower():
                stop_token_ids = [1, 2]
            elif "llama-3.1" in model_name.lower():
                stop_token_ids = [128000, 128040]
            elif "llama-3.2" in model_name.lower():
                stop_token_ids = [128000, 128001, 128008, 128009]
            elif "llama-2" in model_name.lower():
                stop_token_ids = [1, 2]
            elif "mistral" in model_name.lower():
                stop_token_ids = [1, 2]
            elif "glm" in model_name.lower():
                stop_token_ids = [151329, 151336, 151338]
        self.select_tokens = select_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.max_model_len = max_model_len
        self.max_generate_model_len = (
            max_generate_model_len
            if max_generate_model_len is not None
            else max_model_len
        )
        self.model = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
            trust_remote_code=True,
            max_model_len=self.max_generate_model_len,
            enable_lora=enable_lora,
            tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
        )
        self.temperature = temperature
        self.top_p = top_p
        if reward_model == None:
            self.reward_model = self.model
        else:
            self.reward_model = reward_model
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.stop_token_ids = stop_token_ids
        self.stop = stop
        self.select_token_id = [
            self.tokenizer.encode(select_token)[-1] for select_token in select_tokens
        ]

    def get_logis(self, logit_dict: dict, token: int):
        if token not in logit_dict.keys():
            # print('not find token at top20')
            return -1e9
        return logit_dict[token].logprob

    def rewards_predict(self, reward_inputs, select_token_id=None):
        if select_token_id is None:
            select_token_id = self.select_token_id
        inputs = []
        indicis = []
        for i, nodes in enumerate(reward_inputs):
            for node in nodes:
                indicis.append(i)
                inputs.append(node)
        results = self.reward_model.generate(
            inputs,
            SamplingParams(
                top_p=1,
                max_tokens=1,
                logprobs=20,
            ),
            use_tqdm=self.use_tqdm,
        )
        logits_output = [[] for i in range(len(reward_inputs))]
        for i, result in enumerate(results):
            try:
                logit_dict = result.outputs[0].logprobs[0]
                logits = [self.get_logis(logit_dict, id) for id in select_token_id]
            except:
                logits = [1, 1]
            logits_output[indicis[i]].append(logits)
        return logits_output

    def generate_actions(
        self,
        action_inputs,
        n_action=1,
        out_prefix="\n",
        stop=None,
        temperature=None,
        top_p=None,
        top_k=None,
    ):
        if stop is None:
            stop = self.stop
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if top_k is None:
            top_k = self.top_k
        generate_result = self.model.generate(
            action_inputs,
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=self.max_tokens,
                n=n_action,
                top_k=top_k,
                stop=stop,
                stop_token_ids=(self.stop_token_ids),
            ),
            use_tqdm=self.use_tqdm,
        )
        outputs = []
        for result in generate_result:
            for t in result.outputs:
                outputs.append(t.text + out_prefix)
        return outputs

    def generate(self, model_inputs, flags, states, middle_results, n_action: int = 1):
        action_inputs, action_indexs = [], []
        step_inputs, step_indexs = [], []
        reward_inputs, reward_indexs = [], []
        question_inputs, question_indexs = [], []
        for i in range(len(states)):
            if states[i] == None or flags[i] == False:
                continue
            elif states[i] == "Search_End":
                flags[i] = False
            elif "end" in states[i].lower():
                continue
            elif "fast_reward" == states[i]:
                reward_inputs.append(model_inputs[i])
                reward_indexs.append(i)
            elif "get_question" == states[i]:
                question_inputs.append(model_inputs[i])
                question_indexs.append(i)
            elif "get_action" == states[i]:
                action_inputs.append(model_inputs[i])
                action_indexs.append(i)
            elif "step" == states[i]:
                step_inputs.append(model_inputs[i])
                step_indexs.append(i)
        if len(action_inputs) != 0:
            print("generate action")
            actions = self.generate_actions(action_inputs, n_action)
            for i, index in enumerate(action_indexs):
                middle_results[index].action_outputs = actions[
                    i * n_action : (i + 1) * n_action
                ]

        if len(reward_inputs) != 0:
            print("generate rewards")
            logits = self.rewards_predict(reward_inputs)

            for i, index in enumerate(reward_indexs):
                middle_results[index].logits = logits[i]
        if len(step_inputs) != 0:
            print("generate states")
            inputs = []
            for i, t in enumerate(step_inputs):
                inputs.extend(t)
            steps = self.generate_actions(inputs)
            outputs = [[] for i in range(len(step_inputs))]
            k = 0
            for i in range(len(step_inputs)):
                for j in range(len(step_inputs[i])):
                    outputs[i].append(steps[k])
                    k += 1
            for i, index in enumerate(step_indexs):
                middle_results[index].step_outputs = outputs[i]

        if len(question_inputs) != 0:
            print("generate question")
            inputs = []
            for i, t in enumerate(question_inputs):
                inputs.extend(t)
            questions = self.generate_actions(inputs)
            outputs = [[] for i in range(len(question_inputs))]
            k = 0
            for i in range(len(question_inputs)):
                for j in range(len(question_inputs[i])):
                    outputs[i].append(questions[k])
                    k += 1
            for i, index in enumerate(question_indexs):
                middle_results[index].questions = outputs[i]
        gc.collect()


class ChatGenerateModel(GenerateModel):
    def __init__(
        self,
        model_name: str,
        prompt=None,
        max_tokens=768,
        code_topp=0.1,
        code_topk=1024,
        code_temperature=1,
        **kwargs,
    ):
        super().__init__(model_name=model_name, max_tokens=max_tokens, **kwargs)
        self.prompt = prompt
        self.code_topp = code_topp
        self.code_topk = code_topk
        self.code_temperature = code_temperature

    def chat_generate(
        self,
        inputs,
        prefix="",
        n: int = 1,
        out_put_add_prefix: bool = True,
        stop=None,
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
    ):
        """
        根据输入生成聊天回复。

        :param inputs: 用户的输入，可以是单个输入或输入列表。
        :param prefix: 生成文本前添加的前缀。
        :param n: 生成的回复数量。
        :param out_put_add_prefix: 是否在输出中添加前缀。
        :param stop: 停止生成的条件。
        :param temperature: 生成文本的随机性。
        :param top_p: 核采样的比例。
        :param top_k: 从多少个候选词中选择下一个词。
        :return: 生成的聊天回复列表。
        """

        # 确保输入是一个列表
        if not isinstance(inputs[0], list):
            inputs = [inputs]
        # 应用聊天模板并添加生成提示
        chat_inputs = []
        for t in inputs:
            if t[-1]["role"] == "user":
                chat_inputs.append(
                    self.tokenizer.apply_chat_template(
                        t, tokenize=False, add_generation_prompt=True
                    )
                    + prefix
                )
            else:
                chat_inputs.append(
                    self.tokenizer.apply_chat_template(
                        t, tokenize=False, add_generation_prompt=False
                    )[:-11]
                    + prefix
                )

        # 使用类初始化时设置的默认参数值，如果未提供
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if top_k is None:
            top_k = self.top_k

        # 生成回复
        generate_result = self.model.generate(
            chat_inputs,
            SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=self.max_tokens if max_tokens == None else max_tokens,
                n=n,
                top_k=top_k,
                stop=stop,
                stop_token_ids=(self.stop_token_ids),
            ),
            use_tqdm=self.use_tqdm,
        )

        # 处理生成结果，构建输出格式
        outputs = []
        for i, result in enumerate(generate_result):
            for t in result.outputs:
                if inputs[i][-1]["role"] == "user":
                    if out_put_add_prefix:
                        outputs.append(
                            {"role": "assistant", "content": prefix + t.text}
                        )
                    else:
                        outputs.append({"role": "assistant", "content": t.text})
                else:
                    out = inputs[i][-1]["content"] + t.text
                    outputs.append({"role": "assistant", "content": out})

        return outputs

    def generate_code(
        self,
        inputs,
        eval_function=None,
        prefix: str = "```python",
        stop: str = "```",
        flags=None,
        outputs=None,
        iter_num=3,
        show_code=False,
    ):
        """
        生成并执行代码。

        本函数尝试为每个输入生成代码并执行，直到所有输入都成功或尝试次数达到最大值。

        参数:
        - inputs: 一个列表，包含所有需要生成代码的输入。
        - eval_function: 一个函数，用于评估生成的代码。如果为None，则不进行评估。
        - prefix: 代码前缀，默认为'```python'。
        - stop: 代码结束标志，默认为'```'。
        - flags: 一个列表，用于跟踪每个输入的完成状态。如果为None，则初始化为所有输入的True列表。
        - outputs: 一个列表，用于存储每个输入的执行结果。如果为None，则初始化为所有输入的[None,None]列表。

        返回:
        - outputs: 一个列表，包含所有输入的执行结果。
        - flags: 一个列表，表示每个输入是否成功生成并执行代码。
        """

        # 初始化标志列表，用于跟踪每个输入的完成状态
        if flags is None:
            flags = [True] * len(inputs)
        # 深拷贝输入列表，以避免修改原始输入
        inputs = [deepcopy(t) for t in inputs]
        # 初始化输出列表，用于存储每个输入的执行结果
        if outputs is None:
            outputs = [[None, None] for i in range(len(inputs))]
        # 最多尝试生成代码并执行3次
        for iter in range(iter_num):
            # 如果所有输入都已完成，跳出循环
            if not any(flags):
                break
            # 初始化本次循环的代码输入列表和索引列表
            code_inputs = []
            indexs = []
            # 遍历输入列表，将未完成的输入添加到本次循环的代码输入列表中
            for i in range(len(inputs)):
                if flags[i]:
                    code_inputs.append(inputs[i])
                    indexs.append(i)
            # 调用聊天生成函数，生成代码
            code_results = self.chat_generate(
                code_inputs,
                temperature=self.code_temperature,
                top_k=self.code_topk,
                top_p=self.code_topp,
                prefix=prefix,
                stop=stop,
                out_put_add_prefix=False,
            )
            # 遍历生成的代码结果，执行代码并更新输出列表和标志列表
            for result_index, inputs_index in enumerate(indexs):
                code_result, code_input = (
                    code_results[result_index]["content"],
                    code_inputs[result_index],
                )
                code_result = deepcopy(code_result.replace("print(", "scan_to_print = ("))
                code_result = eval_function([code_result, code_input])
                outputs[inputs_index] = code_result
                if code_result[0] is not None:
                    flags[inputs_index] = False
                else:
                    if show_code and iter == iter_num - 1:
                        print(code_result)
                    # 添加修正信息让模型可以修正代码
                    if inputs[inputs_index][-1]["role"] == "user":
                        inputs[inputs_index].append(
                            {
                                "role": "assistant",
                                "content": "```python" + code_result[1][1] + "\n```",
                            }
                        )
                        inputs[inputs_index].append(
                            {
                                "role": "user",
                                "content": "Your code generation contains an error. Please regenerate the code based on the following error message"
                                + code_result[1][0],
                            }
                        )
                    else:
                        initial_inputs = inputs[inputs_index][-1]
                        inputs[inputs_index] = inputs[inputs_index][:-1]
                        inputs[inputs_index].append(
                            {
                                "role": "assistant",
                                "content": "```python" + code_result[1][1] + "\n```",
                            }
                        )
                        inputs[inputs_index].append(
                            {
                                "role": "user",
                                "content": "Your code generation contains an error. Please regenerate the code based on the following error message"
                                + code_result[1][0],
                            }
                        )
                        inputs[inputs_index].append(initial_inputs)
            # 检查是否有输入未完成
            if any(flags):
                print("有%s条数据生成代码失败" % sum(flags))
            else:
                print("所有代码都生成成功")
        return outputs, flags

    def extract_variable(self, dataset):
        """
        从数据集中提取变量并更新数据集。

        该函数根据提供的提示字典生成代码，通过执行生成的代码来识别和提取变量，并将这些变量添加回原始数据集中。

        参数:
        - dataset: 包含需要提取变量的问题的列表。
        - prompt: 包含生成代码所需提示的字典。

        返回:
        - 更新后的数据集，包含提取到的变量信息。
        """
        # 生成输入代码模板，用于后续执行以提取变量
        inputs = [
            self.prompt["get_var_example"]
            + [
                {
                    "role": "user",
                    "content": self.prompt["get_var_prefix"] + t["question"],
                },
                self.prompt["get_var_prompt"],
            ]
            for t in dataset
        ]

        def eval_function(inputs):
            code_result, code_input = inputs
            """
            执行生成的代码并提取变量。

            此函数尝试执行生成的代码片段，并捕获执行过程中定义的变量。

            参数:
            - code_result: 生成的代码结果字符串。

            返回:
            - 成功时返回代码和捕获的变量列表，失败时返回None。
            """
            try:
                local_vars = {}
                code_result = "known_variables = {\n    " + code_result
                exec((code_result + "}"), {}, local_vars)
                known_variables = local_vars.get("known_variables")
                return [code_result + "}", known_variables]
            except Exception as e:
                if "is not defined" in str(e):
                    return [
                        None,
                        [
                            """For the dictionary known_variables, all keys must be of type str, and each value corresponding to the keys must be either of type float or int. Furthermore, no external variables should be called.
We found that your generated code has a key that calls an external variable when defining its value. Please modify it to a form that does not require calling an external variable. If this is not possible, you may remove this key.""",
                            code_result,
                        ],
                    ]
                return [None, [str(e), code_result]]

        # 生成并执行代码以提取变量
        prefix = "```python\nknown_variables = {\n    "
        stop = "}"
        outputs, flags = self.generate_code(
            inputs, prefix=prefix, stop=stop, eval_function=eval_function
        )
        old_topp = self.code_topp
        for _ in range(5):
            if not any(flags):
                break
            print("发现%d条数据没能成功生成代码，尝试重新生成" % sum(flags))
            self.code_topp = min(1, self.code_topp + 0.1)
            outputs, flags = self.generate_code(
                inputs,
                prefix=prefix,
                stop=stop,
                eval_function=eval_function,
                flags=flags,
                outputs=outputs,
            )
        self.code_topp = old_topp
        # 更新数据集以包含提取到的变量
        for i in range(len(outputs)):
            dataset[i]["known_variables"] = outputs[i][1]
            dataset[i]["known_variables_generate"] = outputs[i][0]

        return dataset

    def generate_solve_code(self, inputs, prompt):
        """
        生成并评估代码解决方案，根据给定的输入和提示信息。

        参数:
        - inputs: 问题的输入信息，用于生成代码。
        - prompt: 提示信息，包含如何格式化输出结果的模板。

        返回:
        - exec_codes: 生成的可执行代码列表。
        - codes_responses: 代码执行结果的文本描述列表。
        - param_inputs: 输入参数列表。
        - exec_results: 代码执行结果列表。
        """
        import_prefix = prompt["import_prefix"]

        def eval_function(inputs):
            """
            评估生成的代码是否正确执行，并检查是否存在特定的变量。

            参数:
            - inputs: 包含生成的代码和原始输入信息的元组。

            返回:
            - 如果代码中缺少变量或存在错误，则返回错误信息和原始代码。
            - 否则，返回格式化后的代码和输入输出变量。
            """
            code_result, code_input = deepcopy(inputs)
            local_vars = {}

            code_result = "\nparams_input = {\n    '" + code_result
            try:
                # 尝试执行格式化后的代码
                execute(import_prefix + code_result, local_vars)
            except Exception as e:
                error = str(e)
                try:
                    eval(error)
                    return None, [
                        "Find a dict variable not have a key %s,but use it" % str(e),
                        code_result,
                    ]
                except:
                    pass
                try:
                    # 如果上述尝试失败，则尝试直接执行原始代码
                    if "is not defined" in error:
                        prefix_code = ""
                        for t in code_input:
                            if "```python" in t["content"]:
                                matches = re.findall(
                                    r"```python(.*?)```", t["content"], re.DOTALL
                                )
                                code = matches[0]
                                while " known_variables = {" in code:
                                    code = code.replace(
                                        " known_variables = {", "known_variables = {"
                                    )
                        execute(
                            prefix_code + "\n" + import_prefix + code_result, local_vars
                        )
                        return None, [
                            "your code find a error call:"
                            + str(e)
                            + "\nThe generated code should be able to execute successfully without relying on external variables.",
                            code_result,
                        ]
                    elif "timed out" in error.lower():
                        return None, [
                            "Your code may contain an infinite loop. Please modify your code. Try to avoid using a while loop; you can change it to a for loop or use Python libraries like math, sympy, or scipy to solve your problem.",
                            code_result,
                        ]
                    else:
                        execute(
                            import_prefix + code_result.replace("\n    ", "\n"),
                            local_vars,
                        )
                except Exception as e:
                    error = str(e)
                    if "is not defined" in error:
                        return None, [
                            "your code find a error call:"
                            + str(e)
                            + "\nThe generated code should be able to execute successfully without relying on external variables.",
                            code_result,
                        ]
                    elif "timed out" in error.lower():
                        return None, [
                            "Your code may contain an infinite loop. Please modify your code. Try to avoid using a while loop; you can change it to a for loop or use Python libraries like math, sympy, or scipy to solve your problem.",
                            code_result,
                        ]
                    return None, ["your code find a error call:" + str(e), code_result]

            # 从执行结果中提取输入参数和输出结果
            params_input = local_vars.get("params_input")
            output_result = local_vars.get("output_result")
            if output_result is None:
                return None, [
                    "This code does not have a 'output_result' variable to represent the function's output.You should add the 'output_result' variable in your code to represent the output variables of this problem.",
                    code_result,
                ]

            if not isinstance(output_result, dict):
                return None, [
                    "The variable 'output_result' should be a Python dict class variable",
                    code_result,
                ]

            if output_result is None:
                return None, [
                    "There is no 'output_result' variable in this code to represent the output of the function.",
                    code_result,
                ]
            try:
                for key, value in output_result.items():
                    assert type(value) != bool
                    if type(value) == str:
                        value = re.findall(r"-?\d+\.\d+|-?\d+", value)[0]
                    float(value)
                    output_result[key] = value
            except:
                return None, [
                    "For all values in the dictionary variable 'output_result', they should be either int or float variables. The value '%s' of key '%s' is not an int or float variable.The value's dtype is %s"
                    % (value, key, str(type(value).__name__)),
                    code_result,
                ]
            return [code_result, [params_input, output_result]]

        # 生成可能的代码解决方案
        prefix = "```python\nparams_input = {\n    '"
        code_results, flags = self.generate_code(
            inputs, prefix=prefix, eval_function=eval_function
        )
        old_topp = self.code_topp
        old_topk = self.code_topk
        for _ in range(3):
            if not any(flags):
                break
            print("发现%d条数据没能成功生成代码，尝试重新生成" % sum(flags))
            self.code_topp = min(1, self.code_topp + 0.3)
            self.code_topk = min(max(64, old_topk), self.code_topk + 16)
            code_results, flags = self.generate_code(
                inputs,
                prefix=prefix,
                eval_function=eval_function,
                flags=flags,
                outputs=code_results,
            )

        if any(flags):
            print("Warning:最终发现%d条数据没能成功生成代码!!!!!" % sum(flags))

        self.code_topp = old_topp
        self.code_topk = old_topk
        exec_codes, codes_responses, param_inputs, exec_results = [], [], [], []
        for t in code_results:
            if t[0] is None:
                param_inputs.append(None)
                exec_results.append(None)
                exec_codes.append(None)
                codes_responses.append(None)
            else:
                param_input, exec_result = t[1]
                # 格式化执行结果为文本
                answers = list(exec_result.values())
                if len(answers) == 1:
                    answer = answers[0]
                else:
                    answer = ",".join([str(x) for x in answers])

                exec_result_sentence = prompt["exec_result_prompt"].format(
                    result=str(exec_result).replace("\\", ""), answer=answer
                )
                param_inputs.append(param_input)
                exec_results.append(exec_result)
                exec_codes.append(t[0])
                codes_responses.append(
                    "```python\n" + t[0] + "```\n" + exec_result_sentence
                )
        return exec_codes, codes_responses, param_inputs, exec_results

    def generate(self, model_inputs, flags, states, middle_results, n_action: int):
        """
        根据不同的状态生成相应的输出。

        参数:
        - model_inputs: 模型输入数据。
        - flags: 标志数组，指示是否应处理相应的输入。
        - states: 状态数组，定义了当前每个输入的状态。
        - middle_results: 中间结果数组，用于存储生成的输出。
        - n_action: 动作数量，用于控制生成的动作数量。

        此函数根据输入的状态，将相应的输入数据分配到不同的处理分支中，
        并生成相应的输出，包括动作、奖励、状态和问题。
        """
        # 初始化不同状态下的输入数据和索引列表
        action_inputs, action_indexs = [], []
        step_inputs, step_indexs = [], []
        reward_inputs, reward_indexs = [], []
        question_inputs, question_indexs = [], []

        # 遍历所有输入状态，根据状态对输入进行分类
        for i in range(len(states)):
            if states[i] is None or flags[i] is False:
                # 如果状态为None或标志为False，则跳过当前输入
                continue
            elif states[i] == "Search_End":
                # 如果状态为'Search_End'，则将对应标志设为False
                flags[i] = False
            elif "end" in states[i].lower():
                # 如果状态中包含'end'，则跳过当前输入
                continue
            elif "fast_reward" == states[i]:
                # 如果状态为'fast_reward'，则将输入添加到reward_inputs中
                reward_inputs.append(model_inputs[i])
                reward_indexs.append(i)
            elif "get_question" == states[i]:
                # 如果状态为'get_question'，则将输入添加到question_inputs中
                question_inputs.append(model_inputs[i])
                question_indexs.append(i)
            elif "get_action" == states[i]:
                # 如果状态为'get_action'，则将输入添加到action_inputs中
                action_inputs.append(model_inputs[i])
                action_indexs.append(i)
            elif "step" == states[i]:
                # 如果状态为'step'，则将输入添加到step_inputs中
                step_inputs.append(model_inputs[i])
                step_indexs.append(i)

        # 如果有动作输入，则生成相应数量的动作
        if len(action_inputs) != 0:
            print("generate action")
            actions = self.chat_generate(action_inputs, n=n_action)
            actions = [t["content"] for t in actions]
            for i, index in enumerate(action_indexs):
                middle_results[index].action_outputs = actions[
                    i * n_action : (i + 1) * n_action
                ]

        # 如果有奖励输入，则生成奖励
        if len(reward_inputs) != 0:
            print("generate rewards")
            logits = self.rewards_predict(reward_inputs)
            for i, index in enumerate(reward_indexs):
                middle_results[index].logits = logits[i]

        # 如果有步骤输入，则生成步骤状态
        if len(step_inputs) != 0:
            print("generate states")
            inputs = []
            for i, t in enumerate(step_inputs):
                inputs.extend(t)
            exec_codes, codes_responses, param_inputs, exec_results = (
                self.generate_solve_code(inputs, self.prompt)
            )
            outputs = [[] for i in range(len(step_inputs))]
            k = 0
            for i in range(len(step_inputs)):
                for j in range(len(step_inputs[i])):
                    outputs[i].append(
                        [
                            exec_codes[k],
                            codes_responses[k],
                            param_inputs[k],
                            exec_results[k],
                        ]
                    )
                    k += 1
            # outputs = np.array(outputs)
            for i, index in enumerate(step_indexs):
                middle_results[index].exec_code = [t[0] for t in outputs[i]]
                middle_results[index].step_outputs = [t[1] for t in outputs[i]]
                middle_results[index].para_input = [t[2] for t in outputs[i]]
                middle_results[index].para_output = [t[3] for t in outputs[i]]

        # 如果有问题输入，抛出异常表示不支持此模式
        if len(question_inputs) != 0:
            raise ("not support this mode")

        # 清理垃圾内存
        gc.collect()


class ChatCOTModel(ChatGenerateModel):
    def __init__(
        self,
        model_name,
        action_max_tokens=None,
        action_topp=0,
        action_topk=0,
        action_temperature=None,
        native_rewards_mode=True,
        Qmodel=None,
        Vmodel=None,
        evaluator=None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.Qmodel = Qmodel
        self.Vmodel = Vmodel
        self.evaluator = evaluator
        self.use_Qmodel = True
        self.native_rewards_mode = native_rewards_mode
        self.model_name = model_name
        if action_max_tokens == None:
            action_max_tokens = self.max_tokens
        self.action_max_tokens = action_max_tokens
        if action_temperature == None:
            self.action_temperature = self.temperature
        else:
            self.action_temperature = action_temperature
        if action_topp == 0:
            self.action_topp = self.top_p
        else:
            self.action_topp = action_topp
        if action_topk == 0:
            self.action_topk = self.top_k
        else:
            self.action_topk = action_topk

    def generate_step(self, inputs: list) -> list:
        """
        根据输入数据生成相应的输出步骤。

        该函数首先将输入数据分为两类：特殊问题和普通问题。特殊问题是以特定前缀开始的问题，
        而普通问题是不包含该前缀的其他问题。然后，针对这两类问题分别进行处理和生成回答。
        最后，将生成的回答按照原始输入数据的顺序进行合并和返回。

        参数:
        inputs (list): 包含多个问题序列的列表，每个问题序列是一个字典列表，代表一次对话历史。

        返回:
        list: 包含多个回答序列的列表，每个回答序列是一个字符串列表，代表生成的回答。
        """
        # 初始化最终和普通问题的索引和内容列表
        finnal_indexs = []
        finnal_inputs = []
        normal_inputs = []
        normal_indexs = []

        # 遍历输入数据，对每个对话历史进行分类
        for i, t in enumerate(inputs):
            # 判断是否为特殊问题
            if (
                self.prompt["overall_question_prefix"] in t[-1]["content"]
                and self.prompt["question_postfix"] == "**"
            ):
                if t[-1]["content"][-1] == "\n" and t[-1]["content"][-3:] != "**\n":
                    t[-1]["content"] = t[-1]["content"][:-1] + "**\n"
                finnal_inputs.append(t)
                finnal_indexs.append(i)
            else:
                normal_inputs.append(t)
                normal_indexs.append(i)

        # 如果有特殊问题，进行特殊问题的回答生成
        if len(finnal_inputs) != 0:
            final_outputs = self.chat_generate(finnal_inputs, stop=None)
            # 对生成的回答进行后处理，移除原始问题内容
            for i in range(len(finnal_inputs)):
                final_outputs[i] = final_outputs[i]["content"].replace(
                    finnal_inputs[i][-1]["content"], ""
                )

        # 如果有普通问题，进行普通问题的回答生成
        if len(normal_inputs) != 0:
            normal_outputs = self.chat_generate(normal_inputs, stop=self.stop)
            # 对生成的回答进行后处理，移除原始问题内容并格式化输出
            for i in range(len(normal_outputs)):
                normal_outputs[i] = (
                    normal_outputs[i]["content"].replace(
                        normal_inputs[i][-1]["content"], ""
                    )
                    + "\n\n"
                )

        # 初始化最终输出列表
        outputs = [[] for i in range(len(inputs))]

        # 将普通问题的回答放入最终输出列表的相应位置
        for i, index in enumerate(normal_indexs):
            outputs[index] = normal_outputs[i]

        # 将特殊问题的回答放入最终输出列表的相应位置
        for i, index in enumerate(finnal_indexs):
            outputs[index] = final_outputs[i]

        # 返回最终输出列表
        return outputs

    def generate(self, model_inputs, flags, states, middle_results, n_action: int = 1):
        action_inputs, action_indexs = [], []
        step_inputs, step_indexs = [], []
        reward_inputs, reward_indexs = [], []
        question_inputs, question_indexs = [], []
        revise_inputs, revise_indexs = [], []
        for i in range(len(states)):
            if states[i] == None or flags[i] == False:
                continue
            elif states[i] == "Search_End":
                flags[i] = False
            elif "end" in states[i].lower():
                continue
            elif "fast_reward" == states[i]:
                reward_inputs.append(model_inputs[i])
                reward_indexs.append(i)
            elif "get_question" == states[i]:
                question_inputs.append(model_inputs[i])
                question_indexs.append(i)
            elif "get_action" == states[i]:
                action_inputs.append(model_inputs[i])
                action_indexs.append(i)
            elif "step" == states[i]:
                step_inputs.append(model_inputs[i])
                step_indexs.append(i)
            elif "revise" == states[i]:
                revise_inputs.append(model_inputs[i])
                revise_indexs.append(i)
        if len(action_inputs) != 0:
            print("generate action")
            actions = self.chat_generate(
                action_inputs,
                n=n_action,
                stop=self.prompt["question_postfix"],
                max_tokens=self.action_max_tokens,
                top_p=self.action_topp,
                top_k=self.action_topk,
                temperature=self.action_temperature,
            )
            for i, index in enumerate(action_indexs):
                action_outputs = actions[i * n_action : (i + 1) * n_action]
                middle_results[index].action_outputs = []
                for j in range(len(action_outputs)):
                    try:
                        action_outputs[j] = (
                            action_outputs[j]["content"].replace(
                                action_inputs[i][-1]["content"], ""
                            )
                            + self.prompt["question_postfix"]
                            + "\n"
                        )
                    except:
                        action_outputs[j] = ""
                    while "***" in action_outputs[j]:
                        action_outputs[j] = action_outputs[j].replace("***", "**")
                    if (
                        len(self.tokenizer.encode(action_outputs[j]))
                        < self.action_max_tokens - 10
                        and "🌈" not in action_outputs[j]
                    ):
                        middle_results[index].action_outputs.append(action_outputs[j])
        if len(reward_inputs) != 0:
            print("generate rewards")
            if self.native_rewards_mode or self.Qmodel == None:
                logits = self.rewards_predict(deepcopy(reward_inputs))
            else:
                assert self.Qmodel != None and self.Vmodel != None
                if self.use_Qmodel:
                    self.reward_model = self.Qmodel
                else:
                    self.reward_model = self.Vmodel
                logits = self.rewards_predict(deepcopy(reward_inputs))
                self.use_Qmodel = not self.use_Qmodel
            for i, index in enumerate(reward_indexs):
                middle_results[index].logits = logits[i]

        if len(step_inputs) != 0:
            print("generate states")
            inputs = []
            for i, t in enumerate(step_inputs):
                inputs.extend(t)
            steps = self.generate_step(inputs)
            outputs = [[] for i in range(len(step_inputs))]
            k = 0
            for i in range(len(step_inputs)):
                for j in range(len(step_inputs[i])):
                    outputs[i].append(steps[k])
                    k += 1
            for i, index in enumerate(step_indexs):
                middle_results[index].step_outputs = outputs[i]
        if len(revise_inputs) != 0:
            print("revise answer")
            inputs = []
            for i, t in enumerate(revise_inputs):
                inputs.extend(t)
            revise_outputs = self.chat_generate(
                inputs,
                n=1,
                stop=None,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                temperature=self.temperature,
            )
            outputs = [[] for i in range(len(revise_inputs))]
            k = 0
            for i in range(len(revise_inputs)):
                for j in range(len(revise_inputs[i])):
                    outputs[i].append(revise_outputs[k]["content"])
                    k += 1
            for i, index in enumerate(revise_indexs):
                middle_results[index].revise_result = outputs[i]
        if len(question_inputs) != 0:
            raise ("not support this mode")

    def rewards_predict(self, reward_inputs):
        if self.native_rewards_mode:
            return super().rewards_predict(reward_inputs)
        yes_inputs = []
        no_inputs = []
        indicis = []
        for i, nodes in enumerate(reward_inputs):
            for node in nodes:
                indicis.append(i)
                yes_inputs.append(node + self.select_tokens[0])
                no_inputs.append(node + self.select_tokens[1])
        yes_outputs = self.reward_model.generate(
            yes_inputs,
            SamplingParams(max_tokens=1, prompt_logprobs=20),
            use_tqdm=self.use_tqdm,
        )
        no_outputs = self.reward_model.generate(
            no_inputs,
            SamplingParams(max_tokens=1, prompt_logprobs=20),
            use_tqdm=self.use_tqdm,
        )
        logits_output = [[] for i in range(len(reward_inputs))]
        for i, result in enumerate(no_outputs):
            try:
                logits = [
                    self.get_logis(
                        yes_outputs[i].prompt_logprobs[-1], self.select_token_id[0]
                    ),
                    self.get_logis(
                        no_outputs[i].prompt_logprobs[-1], self.select_token_id[1]
                    ),
                ]
            except:
                logits = [-1e9, 1]
            logits_output[indicis[i]].append(logits)

        return logits_output


class ChatCodeModel(ChatCOTModel):
    def extract_variable(self, dataset, initial_prompt):
        inputs = []
        inital_example = []
        for t in initial_prompt["interactive_examples"]:
            inital_example.extend(t)
        for t in dataset:
            inputs.append(
                inital_example
                + [
                    initial_prompt["instruction"],
                    {"role": "user", "content": t["question"]},
                    {"role": "assistant", "content": initial_prompt["answer_prefix"]},
                ]
            )

        def eval_function(inputs):
            code_result, code_input = inputs
            code_result = code_result.replace("```python", "")
            """
            执行生成的代码并提取变量。

            此函数尝试执行生成的代码片段，并捕获执行过程中定义的变量。

            参数:
            - code_result: 生成的代码结果字符串。

            返回:
            - 成功时返回代码和捕获的变量列表，失败时返回None。
            """
            try:
                local_vars = {}
                execute(code_result, local_vars)
                return [
                    code_result,
                    code_result.replace(initial_prompt["answer_prefix"], ""),
                ]
            except Exception as e:
                if "= None" in code_result:
                    return [
                        code_result,
                        [
                            "The defined variable should not be assigned to None",
                            code_result,
                        ],
                    ]
                return [None, [str(e), code_result]]

        prefix = ""
        stop = initial_prompt["stop_token"]
        outputs, flags = super(ChatCodeModel, self).generate_code(
            inputs, prefix=prefix, stop=stop, eval_function=eval_function
        )
        old_topp = self.code_topp
        old_topk = self.code_topk
        for _ in range(5):
            if not any(flags):
                break
            print("发现%d条数据没能成功生成代码，尝试重新生成" % sum(flags))
            self.code_topp = min(1, self.code_topp + 0.1)
            self.code_topk = self.code_topk + 16
            outputs, flags = super().generate_code(
                inputs,
                prefix=prefix,
                stop=stop,
                eval_function=eval_function,
                flags=flags,
                outputs=outputs,
            )
        self.code_topp = old_topp
        self.code_topk = old_topk

        inital_variable = [{"inital_variable": t[0]} for t in outputs]
        return inital_variable

    def generate_step(self, inputs: list) -> list:
        """
        根据输入数据生成相应的输出步骤。

        该函数首先将输入数据分为两类：特殊问题和普通问题。特殊问题是以特定前缀开始的问题，
        而普通问题是不包含该前缀的其他问题。然后，针对这两类问题分别进行处理和生成回答。
        最后，将生成的回答按照原始输入数据的顺序进行合并和返回。

        参数:
        inputs (list): 包含多个问题序列的列表，每个问题序列是一个字典列表，代表一次对话历史。

        返回:
        list: 包含多个回答序列的列表，每个回答序列是一个字符串列表，代表生成的回答。
        """
        # 初始化最终和普通问题的索引和内容列表
        finnal_indexs = []
        finnal_inputs = []
        normal_inputs = []
        normal_indexs = []

        # 遍历输入数据，对每个对话历史进行分类
        for i, t in enumerate(inputs):
            # 判断是否为特殊问题
            if (
                self.prompt["overall_question_prefix"][:-1].lower()
                in t[-1]["content"].lower()
            ):
                finnal_inputs.append(t)
                finnal_indexs.append(i)
            else:
                normal_inputs.append(t)
                normal_indexs.append(i)

        # 如果有特殊问题，进行特殊问题的回答生成
        if len(finnal_inputs) != 0:
            final_outputs = self.generate_code(finnal_inputs, stop="```")
            # 对生成的回答进行后处理，移除原始问题内容
            for i in range(len(finnal_inputs)):
                final_outputs[i] = final_outputs[i]["content"].replace(
                    finnal_inputs[i][-1]["content"], ""
                )

        # 如果有普通问题，进行普通问题的回答生成
        if len(normal_inputs) != 0:
            normal_outputs = self.generate_code(normal_inputs, stop=self.stop + ["```"])
            # 对生成的回答进行后处理，移除原始问题内容并格式化输出
            for i in range(len(normal_outputs)):
                normal_outputs[i] = (
                    normal_outputs[i]["content"].replace(
                        normal_inputs[i][-1]["content"], ""
                    )
                    + "\n\n"
                )

        # 初始化最终输出列表
        outputs = [[] for i in range(len(inputs))]

        # 将普通问题的回答放入最终输出列表的相应位置
        for i, index in enumerate(normal_indexs):
            outputs[index] = normal_outputs[i]

        # 将特殊问题的回答放入最终输出列表的相应位置
        for i, index in enumerate(finnal_indexs):
            outputs[index] = final_outputs[i]

        # 返回最终输出列表
        return outputs

    def generate_code(self, inputs, stop, iter_num=5):
        def eval_function(inputs):
            code_result, code_input = inputs
            code_result = code_input[-1]["content"] + code_result.replace(
                code_input[-1]["content"], ""
            ).replace("```python", "").replace(
                "\n" + self.prompt["overall_question_prefix"],
                "\n#" + self.prompt["overall_question_prefix"],
            ).replace("##", "#")

            try:
                local_vars = {}
                execute(code_result, local_vars)
                if (
                    self.prompt["overall_question_prefix"] in code_result
                    and local_vars.get("result") == None
                ):
                    return [
                        None,
                        [
                            'The "%s" appears in the code, indicating that the problem should be solved. You should follow the requirements of the system and write the result into a result variable. But there is no such result variable in your code.'
                            % self.prompt["overall_question_prefix"],
                            code_result,
                        ],
                    ]

                code = code_result.replace(code_input[-1]["content"], "")
                execute_result = ""
                flag = True
                for name, value in local_vars.items():
                    if name in code:
                        flag = False
                        execute_result += "%s:%s(%s);" % (
                            str(name),
                            str(value),
                            str(type(value)),
                        )
                execute_result = (
                    '\n"""The running status of existing variables:\n%s\n"""\n'
                    % execute_result
                )
                if flag:
                    return [code_result, code_result]
                return [code_result + execute_result, code_result + execute_result]
            except Exception as e:
                return [None, [str(e), code_result]]

        prefix = ""
        outputs, flags = super(ChatCodeModel, self).generate_code(
            inputs, prefix=prefix, stop=stop, eval_function=eval_function
        )
        old_topp = self.code_topp
        old_topk = self.code_topk
        for iter in range(iter_num):
            if not any(flags):
                break
            print("发现%d条数据没能成功生成代码，尝试重新生成" % sum(flags))
            self.code_topp = min(1, self.code_topp + 0.1)
            self.code_topk = self.code_topk + old_topk
            outputs, flags = super().generate_code(
                inputs,
                prefix=prefix,
                stop=stop,
                show_code=iter == iter_num - 1,
                eval_function=eval_function,
                flags=flags,
                outputs=outputs,
            )
        self.code_topp = old_topp
        self.code_topk = old_topk
        for i in range(len(outputs)):
            outputs[i] = outputs[i][0]
            if outputs[i] != None:
                outputs[i] = outputs[i].replace(inputs[i][-1]["content"], "")
                while "\n\n\n" in outputs[i]:
                    outputs[i] = outputs[i].replace("\n\n\n", "\n\n")
            else:
                outputs[i] = (
                    "\n#This question was not successfully answered, please correct it and propose a new, more reasonable question.\n"
                )
        return [{"content": outputs[i]} for i in range(len(outputs))]


class PRMChatCOTModel(ChatCOTModel):
    def __init__(
        self,
        model_name,
        reward_model_name,
        reward_model_gpu_memory_utilization,
        reward_token="<extra_0>",
        native_rewards_mode=False,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name, native_rewards_mode=native_rewards_mode, **kwargs
        )
        if native_rewards_mode == False:
            self.reward_token = reward_token
            self.reward_model = LLM(
                model=reward_model_name,
                gpu_memory_utilization=reward_model_gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=kwargs["max_model_len"],
                task="reward",
                enable_prefix_caching=True,
            )

    def rewards_predict(self, reward_inputs):
        if self.native_rewards_mode:
            for i in range(len(reward_inputs)):
                reward_inputs[i] = [
                    self.prompt["useful_examples_prefix"]
                    % (t[-2]["content"], t[-1]["content"])
                    for t in reward_inputs[i]
                ]
            return super().rewards_predict(reward_inputs)
        inputs = []
        indicis = []
        for i, nodes in enumerate(reward_inputs):
            for node in nodes:
                indicis.append(i)
                inputs.append(
                    self.tokenizer.apply_chat_template(
                        node, tokenize=False, add_generation_prompt=True
                    )[:-11]
                    + self.reward_token
                )
        reward_outputs = self.reward_model.encode(
            inputs,
            use_tqdm=self.use_tqdm,
        )
        logits_output = [[] for i in range(len(reward_inputs))]
        for i, result in enumerate(reward_outputs):
            reward = result.outputs.data.numpy()[0]
            logits = np.log(reward)[::-1]
            logits_output[indicis[i]].append(logits)
        return logits_output


class DeepMCTSModel(PRMChatCOTModel):
    def generate_step(self, inputs: list) -> list:
        """
        根据输入数据生成相应的输出步骤。

        该函数首先将输入数据分为两类：特殊问题和普通问题。特殊问题是以特定前缀开始的问题，
        而普通问题是不包含该前缀的其他问题。然后，针对这两类问题分别进行处理和生成回答。
        最后，将生成的回答按照原始输入数据的顺序进行合并和返回。

        参数:
        inputs (list): 包含多个问题序列的列表，每个问题序列是一个字典列表，代表一次对话历史。

        返回:
        list: 包含多个回答序列的列表，每个回答序列是一个字符串列表，代表生成的回答。
        """
        # 初始化最终和普通问题的索引和内容列表
        finnal_indexs = []
        finnal_inputs = []
        normal_inputs = []
        normal_indexs = []
        code_inputs = []
        code_indexs = []
        # 遍历输入数据，对每个对话历史进行分类
        for i, mopdel_input in enumerate(inputs):
            t, action = mopdel_input
            # 判断是否为特殊问题
            if action in self.prompt["summar_prompt"]:
                finnal_inputs.append(t)
                finnal_indexs.append(i)
            elif action in self.prompt["code_actions"]:
                code_inputs.append(t)
                code_indexs.append(i)
            else:
                normal_inputs.append(t)
                normal_indexs.append(i)

        # 如果有特殊问题，进行特殊问题的回答生成
        if len(finnal_inputs) != 0:
            print("summary step")
            final_outputs = self.chat_generate(
                finnal_inputs, stop=None, prefix=self.prompt["prefix"]
            )
            # 对生成的回答进行后处理，移除原始问题内容
            for i in range(len(finnal_inputs)):
                final_outputs[i] = final_outputs[i]["content"].replace(
                    finnal_inputs[i][-1]["content"], ""
                )

        # 如果有普通问题，进行普通问题的回答生成
        if len(normal_inputs) != 0:
            print("normal  step")
            normal_outputs = self.chat_generate(
                normal_inputs,
                stop=self.prompt["stop"].replace("\n", ""),
                prefix=self.prompt["prefix"],
            )
            # 对生成的回答进行后处理，移除原始问题内容并格式化输出
            for i in range(len(normal_outputs)):
                normal_outputs[i] = (
                    normal_outputs[i]["content"].replace(
                        normal_inputs[i][-1]["content"], ""
                    )
                    + "\n"
                    + self.prompt["stop"]
                )
        if len(code_inputs) != 0:
            print("code step")
            code_outputs = self.generate_code(code_inputs)
        # 初始化最终输出列表
        outputs = [[] for i in range(len(inputs))]

        # 将普通问题的回答放入最终输出列表的相应位置
        for i, index in enumerate(normal_indexs):
            outputs[index] = normal_outputs[i]

        # 将特殊问题的回答放入最终输出列表的相应位置
        for i, index in enumerate(finnal_indexs):
            outputs[index] = final_outputs[i]

        for i, index in enumerate(code_indexs):
            outputs[index] = code_outputs[i]
        # 返回最终输出列表
        return outputs

    def generate_code(self, inputs):
        prefix = self.prompt["code_prefix"]
        stop = "```"

        def eval_function(inputs):
            code_result, code_input = inputs
            code = code_result.replace(code_input[-1]["content"], "")
            try:
                local_vars = {}
                execute(code, local_vars)
                execute_result = ""
                flag = True
                for name, value in local_vars.items():
                    if name in code and str(type(value)) != "<class 'function'>":
                        flag = False
                        execute_result += "%s:%s(%s);" % (
                            str(name),
                            str(value),
                            str(type(value)),
                        )
                if flag:
                    execute_result = "\nThis  code execute fail\n"
                else:
                    execute_result = (
                        "\nThe running status of existing variables:\n%s\n"
                        % execute_result
                    )
                code_result = (
                    prefix + code + "\n" + self.prompt["code_stop"] + execute_result
                )
                return [code_result, code_result]
            except Exception as e:
                return [None, [str(e), code]]

        outputs, flags = super(DeepMCTSModel, self).generate_code(
            inputs, prefix=prefix, stop=stop, eval_function=eval_function
        )
        for i in range(len(outputs)):
            if not flags[i]:
                outputs[i] = outputs[i][0]
                while "\n\n\n" in outputs[i]:
                    outputs[i] = outputs[i].replace("\n\n\n", "\n\n")
            else:
                outputs[i] = outputs[i][1][-1]
        return outputs


# 1和2的区别是我在代码里加了超时prompt，其他是一模一样的，难题经常会写死循环，所以要加上这个
class DeepMCTSModel2(DeepMCTSModel):
    def generate_code(self, inputs):
        prefix = self.prompt["code_prefix"]
        stop = "```"

        def eval_function(inputs):
            code_result, code_input = inputs
            code = code_result.replace(code_input[-1]["content"], "")
            try:
                local_vars = {}
                execute(code, local_vars)
                execute_result = ""
                flag = True
                for name, value in local_vars.items():
                    if name in code and str(type(value)) != "<class 'function'>":
                        flag = False
                        execute_result += "%s:%s(%s);" % (
                            str(name),
                            str(value),
                            str(type(value)),
                        )
                if flag:
                    execute_result = "\nThis  code execute fail\n"
                else:
                    execute_result = (
                        "\nThe running status of existing variables:\n%s\n"
                        % execute_result
                    )
                code_result = (
                    prefix + code + "\n" + self.prompt["code_stop"] + execute_result
                )
                return [code_result, code_result]
            except Exception as e:
                if "timed out" in str(e).lower():
                    return None, [
                        "Your code may contain an infinite loop. Please modify your code. Try to avoid using a while loop; you can change it to a for loop or use Python libraries like math, sympy, or scipy to solve your problem.",
                        code_result,
                    ]
                return [None, [str(e), code]]

        outputs, flags = super(DeepMCTSModel, self).generate_code(
            inputs, prefix=prefix, stop=stop, eval_function=eval_function
        )
        for i in range(len(outputs)):
            if not flags[i]:
                outputs[i] = outputs[i][0]
                while "\n\n\n" in outputs[i]:
                    outputs[i] = outputs[i].replace("\n\n\n", "\n\n")
            else:
                outputs[i] = outputs[i][1][-1]
        return outputs


codes = []


class DeepMCTSModel3(DeepMCTSModel):
    def generate_code(self, inputs):
        prefix = self.prompt["code_prefix"]
        stop = "```"

        def eval_function(inputs):
            global codes
            code_result, code_input = inputs
            code = code_result.replace(code_input[-1]["content"], "")

            if "jax" in code or "tensorflow" in code:
                return None, [
                    "Your code should not use neural network libraries like JAX or TensorFlow.",
                    code_result,
                ]
            elif "matplotlib" in code:
                return None, [
                    "The visualization code of matplotlib is of no help in solving this problem. Please write a new code. Don't use the matplotlib library.",
                    code_result,
                ]
            try:
                local_vars = {}
                execute(code, local_vars)
                codes.append(code)
                execute_result = ""
                flag = True
                for name, value in local_vars.items():
                    value_type = str(type(value))
                    old_type = type(value)
                    if (
                        name in code
                        and value_type != "<class 'function'>"
                        and "<class 'module'>" not in value_type
                    ):
                        flag = False
                        if "float" in value_type:
                            value = round(value, 4)
                        if (
                            "list" in value_type
                            or "tuple" in value_type
                            or "array" in value_type
                        ):
                            try:
                                value = np.array(value)
                                value = old_type(value)(np.round_(value, 4))
                            except:
                                if "list" in value_type:
                                    value = [
                                        round(x, 4) if isinstance(x, float) else x
                                        for x in value
                                    ]
                                elif "tuple" in value_type:
                                    value = tuple(
                                        [
                                            round(x, 4) if isinstance(x, float) else x
                                            for x in value
                                        ]
                                    )
                        execute_result += "%s:%s;" % (str(name), str(value))
                if flag:
                    execute_result = "\nThis  code execute fail\n"
                else:
                    execute_result = (
                        "\nThe running status of existing variables:\n%s\n"
                        % execute_result
                    )
                code_result = (
                    prefix + code + "\n" + self.prompt["code_stop"] + execute_result
                )
                return [code_result, code_result]
            except Exception as e:
                if "is not defined" in str(e).lower():
                    return None, [
                        "your code find a error call:"
                        + str(e)
                        + "\nThe generated code should be able to execute successfully without relying on external variables.If you have used these variables before, please rewrite them again.",
                        code_result,
                    ]

                if "timed out" in str(e).lower():
                    return None, [
                        "Your code may contain an infinite loop. Please modify your code. Try to avoid using a while loop; you can change it to a for loop or use Python libraries like math, sympy, or scipy to solve your problem.",
                        code_result,
                    ]
                return [None, [str(e), code]]

        outputs, flags = super(DeepMCTSModel, self).generate_code(
            inputs, prefix=prefix, stop=stop, eval_function=eval_function
        )
        for i in range(len(outputs)):
            if not flags[i]:
                outputs[i] = outputs[i][0]
                while "\n\n\n" in outputs[i]:
                    outputs[i] = outputs[i].replace("\n\n\n", "\n\n")
            else:
                outputs[i] = (
                    outputs[i][1][-1] + "This code find error:\n" + outputs[i][1][0]
                )
        return outputs

    def rewards_predict(self, reward_inputs):
        # 主要是加了个长度限制，不然会炸
        if self.native_rewards_mode:
            return super().rewards_predict(reward_inputs)
        inputs = []
        indicis = []
        for i, nodes in enumerate(reward_inputs):
            for node in nodes:
                indicis.append(i)
                token = self.tokenizer.apply_chat_template(
                    node, tokenize=False, add_generation_prompt=True
                )[:-11]
                token = self.tokenizer.decode(
                    self.tokenizer.encode(token)[-self.max_model_len + 3 :]
                )
                inputs.append(token + self.reward_token)
        reward_outputs = self.reward_model.encode(
            inputs,
            use_tqdm=self.use_tqdm,
        )
        logits_output = [[] for i in range(len(reward_inputs))]
        for i, result in enumerate(reward_outputs):
            reward = result.outputs.data.numpy()[0]
            logits = np.log(reward)[::-1]
            logits_output[indicis[i]].append(logits)
        return logits_output


import multiprocessing


def execute_code(code):
    local_vars = {}
    exec(code, local_vars)
    local_vars.pop("__builtins__")
    pop_list = []
    for key, value in local_vars.items():
        if "<function" in str(value) or "<module" in str(value):
            pop_list.append(key)
    for key in pop_list:
        local_vars.pop(key)
    return local_vars


def run_with_timeout(code, timeout=1):
    pool = multiprocessing.Pool(processes=1)
    try:
        result = pool.apply_async(execute_code, (code,))
        return result.get(timeout=timeout)
    except multiprocessing.TimeoutError:
        pool.terminate()
        raise TimeoutError
    finally:
        pool.close()
        pool.join()


class DeepMCTSModel4(DeepMCTSModel3):
    def generate_code(self, inputs):
        prefix = self.prompt["code_prefix"]
        stop = "```"

        def eval_function(inputs):
            global codes
            code_result, code_input = inputs
            code = code_result.replace(code_input[-1]["content"], "")

            if "jax" in code or "tensorflow" in code:
                return None, [
                    "Your code should not use neural network libraries like JAX or TensorFlow.",
                    code_result,
                ]
            elif "matplotlib" in code:
                return None, [
                    "The visualization code of matplotlib is of no help in solving this problem. Please write a new code. Don't use the matplotlib library.",
                    code_result,
                ]
            try:
                local_vars = run_with_timeout(code, timeout=2)
                codes.append(code)
                execute_result = ""
                flag = True
                for name, value in local_vars.items():
                    value_type = str(type(value))
                    old_type = type(value)
                    if (
                        name in code
                        and value_type != "<class 'function'>"
                        and "<class 'module'>" not in value_type
                    ):
                        flag = False
                        if "float" in value_type:
                            value = round(value, 4)
                        if (
                            "list" in value_type
                            or "tuple" in value_type
                            or "array" in value_type
                        ):
                            try:
                                value = np.array(value)
                                value = old_type(value)(np.round_(value, 4))
                            except:
                                if "list" in value_type:
                                    value = [
                                        round(x, 4) if isinstance(x, float) else x
                                        for x in value
                                    ]
                                elif "tuple" in value_type:
                                    value = tuple(
                                        [
                                            round(x, 4) if isinstance(x, float) else x
                                            for x in value
                                        ]
                                    )
                        execute_result += "%s:%s;" % (str(name), str(value))
                if flag:
                    execute_result = "\nThis  code execute fail\n"
                else:
                    execute_result = (
                        "\nThe running status of existing variables:\n%s\n"
                        % execute_result
                    )
                code_result = (
                    prefix + code + "\n" + self.prompt["code_stop"] + execute_result
                )
                return [code_result, code_result]
            except TimeoutError:
                if "sympy" in code:
                    return None, [
                        "Please ensure that all sympy calculations can be completed within 2 seconds, and the number of polynomials should not exceed 6.",
                        code_result,
                    ]
                return None, [
                    "Your code may contain an infinite loop. Please modify your code. Try to avoid using a while loop; you can change it to a for loop or use Python libraries like math, sympy, or scipy to solve your problem.",
                    code_result,
                ]
            except Exception as e:
                if "is not defined" in str(e).lower():
                    return None, [
                        "your code find a error call:"
                        + str(e)
                        + "\nThe generated code should be able to execute successfully without relying on external variables.If you have used these variables before, please rewrite them again.",
                        code_result,
                    ]

                if "timed out" in str(e).lower():
                    return None, [
                        "Your code may contain an infinite loop. Please modify your code. Try to avoid using a while loop; you can change it to a for loop or use Python libraries like math, sympy, or scipy to solve your problem.",
                        code_result,
                    ]
                return [None, [str(e), code]]

        outputs, flags = super(DeepMCTSModel, self).generate_code(
            inputs, prefix=prefix, stop=stop, eval_function=eval_function
        )
        for i in range(len(outputs)):
            if not flags[i]:
                outputs[i] = outputs[i][0]
                while "\n\n\n" in outputs[i]:
                    outputs[i] = outputs[i].replace("\n\n\n", "\n\n")
            else:
                outputs[i] = (
                    outputs[i][1][-1] + "This code find error:\n" + outputs[i][1][0]
                )
        return outputs

class DeepMCTSModel5(DeepMCTSModel3):
    def generate_code(self, inputs):
        prefix = self.prompt["code_prefix"]
        stop = "```"

        def eval_function(inputs):
            global codes
            code_result, code_input = inputs
            code = code_result.replace(code_input[-1]["content"], "")

            if "jax" in code or "tensorflow" in code:
                return None, [
                    "Your code should not use neural network libraries like JAX or TensorFlow.",
                    code_result,
                ]
            elif "matplotlib" in code:
                return None, [
                    "The visualization code of matplotlib is of no help in solving this problem. Please write a new code. Don't use the matplotlib library.",
                    code_result,
                ]
            try:
                local_vars = run_with_timeout(code, timeout=2)
                codes.append(code)
                execute_result = ""
                flag = True
                for name, value in local_vars.items():
                    value_type = str(type(value))
                    old_type = type(value)
                    if (
                        name in code
                        and value_type != "<class 'function'>"
                        and "<class 'module'>" not in value_type
                    ):
                        flag = False
                        if "float" in value_type:
                            value = round(value, 4)
                        if (
                            "list" in value_type
                            or "tuple" in value_type
                            or "array" in value_type
                        ):
                            try:
                                value = np.array(value)
                                value = old_type(value)(np.round_(value, 4))
                            except:
                                if "list" in value_type:
                                    value = [
                                        round(x, 4) if isinstance(x, float) else x
                                        for x in value
                                    ]
                                elif "tuple" in value_type:
                                    value = tuple(
                                        [
                                            round(x, 4) if isinstance(x, float) else x
                                            for x in value
                                        ]
                                    )
                        execute_result += "%s:%s;" % (str(name), str(value))
                if flag:
                    execute_result = "\nThis  code execute fail\n"
                else:
                    execute_result = (
                        "\nThe running status of existing variables:\n%s\n"
                        % execute_result
                    )
                code_result = (
                    prefix + code + "\n" + self.prompt["code_stop"] + execute_result
                )
                return [code_result, code_result]
            except TimeoutError:
                if "sympy" in code:
                    return None, [
                        "Please ensure that all sympy calculations can be completed within 2 seconds, and the number of polynomials should not exceed 6.",
                        code_result,
                    ]
                return None, [
                    "Your code may contain an infinite loop. Please modify your code. Try to avoid using a while loop; you can change it to a for loop or use Python libraries like math, sympy, or scipy to solve your problem.",
                    code_result,
                ]
            except Exception as e:
                if "is not defined" in str(e).lower():
                    return None, [
                        "your code find a error call:"
                        + str(e)
                        + "\nThe generated code should be able to execute successfully without relying on external variables.If you have used these variables before, please rewrite them again.",
                        code_result,
                    ]

                if "timed out" in str(e).lower():
                    return None, [
                        "Your code may contain an infinite loop. Please modify your code. Try to avoid using a while loop; you can change it to a for loop or use Python libraries like math, sympy, or scipy to solve your problem.",
                        code_result,
                    ]
                return [None, [str(e), code]]

        outputs, flags = super(DeepMCTSModel, self).generate_code(
            inputs, prefix=prefix, stop=stop, eval_function=eval_function
        )
        for i in range(len(outputs)):
            if not flags[i]:
                outputs[i] = outputs[i][0]
                while "\n\n\n" in outputs[i]:
                    outputs[i] = outputs[i].replace("\n\n\n", "\n\n")
            else:
                outputs[i] = (
                    outputs[i][1][-1] + "This code find error:\n" + outputs[i][1][0]
                )
        return outputs
