import io
from typing import NamedTuple, TypedDict
from collections import defaultdict
from reasoners import WorldModel
from reasoners.base import Example
import numpy as np
from reasoners.algorithm.mcts import MiddleResult


class SubResult(NamedTuple):
    sub_question: str
    sub_answer: str
    confidence: float


GSM8kState = list[SubResult]
GSM8kAction = str
GSM8kExample = str
from copy import deepcopy


class GSM8kPromptDict(TypedDict):
    instruction: str
    interactive_examples: list[str]
    useful_examples: list[str]
    question_prefix: str
    subquestion_prefix: str
    overall_question_prefix: str
    answer_prefix: str


class GSM8kWorldModel(WorldModel[GSM8kState, GSM8kAction, GSM8kExample]):
    """
    GSM8k World Model
    State: [[sub_question_1, sub_answer_1, confidence_1], [sub_question_2, sub_answer_2, confidence_2], ...]
    Action: sub_question
    """

    def __init__(
        self,
        n_iters=None,
        base_model=None,
        n_confidence=8,
        batch_size=2,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        early_stop_base=None,
        retrieve_answer=None,
        early_stop_threshold=1.0,
    ) -> None:
        super().__init__()
        self.n_iters = n_iters
        self.base_model = base_model
        self.batch_size = batch_size
        self.n_confidence = n_confidence
        self.temperature = temperature
        self.early_stop_base = (
            early_stop_base if early_stop_base is not None else n_confidence
        )
        self.early_stop_threshold = early_stop_threshold
        self.prompt_examples = ""
        self.n_shots = 0
        self.top_k = top_k
        self.top_p = top_p
        from utils import gsm8k_utils

        self.utils = gsm8k_utils
        if retrieve_answer is None:
            self.retrieve_answer = self.utils.retrieve_answer
        else:
            self.retrieve_answer = retrieve_answer

    def update_example(self, example: Example, prompt) -> None:
        super().update_example(example, prompt)
        assert prompt is not None
        self.prompt = prompt
        with io.StringIO() as f:
            f.write(self.prompt["instruction"] + "\n\n")
            for idx, example in enumerate(self.prompt["interactive_examples"]):
                f.write(example.format(idx=idx + 1) + "\n\n")
            self.n_shots = len(self.prompt["interactive_examples"])
            self.prompt_examples = f.getvalue()

    def init_state(self) -> list:
        return []

    def get_step_model_input(self, state: GSM8kState, action: GSM8kAction):
        with io.StringIO() as f:
            f.write(self.prompt_examples)
            f.write(
                self.prompt["question_prefix"].format(
                    idx=self.n_shots + 1, question=self.example
                )
                + "\n"
            )
            for idx, t in enumerate(state):
                q, a = t[:2]
                f.write(
                    self.prompt["subquestion_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + q
                    + "\n"
                )
                f.write(
                    self.prompt["answer_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + a
                    + "\n"
                )
            f.write(
                self.prompt["subquestion_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=len(state) + 1
                )
                + " "
                + action
                + "\n"
            )
            f.write(
                self.prompt["answer_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=len(state) + 1
                )
            )
            model_input = f.getvalue()
        return model_input

    def get_step_inputs(
        self, state: GSM8kState, action: GSM8kAction
    ) -> tuple[GSM8kState, dict]:
        state = state.copy()

        model_input = self.get_step_model_input(state, action)
        num = 0
        for start1 in range(0, self.n_confidence, self.early_stop_base):
            stop1 = min(start1 + self.early_stop_base, self.n_confidence)

            for start in range(start1, stop1, self.batch_size):
                stop = min(start + self.batch_size, stop1)
                num += stop - start
        return num, state, model_input

    def get_step_outputs(self, outputs, state, action, midlle_result: MiddleResult):
        """
        根据模型的输出，更新状态并返回最可能的答案及信心值。

        参数:
        - outputs: 模型在一步内产生的所有输出。
        - state: 与环境交互的状态，将在此方法中更新。
        - action: 执行的动作，用于更新状态。
        - midlle_result: 中间结果，目前未使用。

        返回:
        - state: 更新后的状态，包括最新的子结果。
        - aux: 辅助信息，包含答案的信心值。
        """
        # 初始化一个字典，用于将答案映射到思考列表
        answer_dict = defaultdict(list)
        result = ""

        # 遍历所有输出，提取答案并存储到answer_dict中
        for output in outputs:
            if output is None:
                continue
            result = output.strip()
            answer = self.retrieve_answer(result)
            answer_dict[answer].append(result)

        # 根据出现次数对答案进行排序
        sorted_answer_dict = sorted(
            answer_dict.items(), key=lambda p: len(p[1]), reverse=True
        )

        # 如果没有找到任何答案，发出警告并回退到选择最后一个输出作为答案
        if len(answer_dict) == 0:
            print("Warning: no answer found")
            confidence, answer = 0, result  # 没有找到合理的答案。回退到选择最后一个响应
        else:
            # 再次排序答案字典，以便获取出现次数最多的答案
            sorted_answer_dict = sorted(
                answer_dict.items(), key=lambda p: len(p[1]), reverse=True
            )
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            max_len = len(max_answer_output_list)

            # 选择出现次数最多的答案中第一次出现的答案，并计算信心值
            answer = max_answer_output_list[0]  # 这里我们简单选择答案第一次出现的情况
            confidence = max_len / sum(len(v) for v in answer_dict.values())

        # 更新状态，包括动作、答案和信心值
        state.append(SubResult(action, answer, confidence))
        aux = {"confidence": confidence}
        return state, aux

    def step(self, state: GSM8kState, action: GSM8kAction) -> tuple[GSM8kState, dict]:
        state = state.copy()

        with io.StringIO() as f:
            f.write(self.prompt_examples)
            f.write(
                self.prompt["question_prefix"].format(
                    idx=self.n_shots + 1, question=self.example
                )
                + "\n"
            )
            for idx, (q, a, _) in enumerate(state):
                f.write(
                    self.prompt["subquestion_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + q
                    + "\n"
                )
                f.write(
                    self.prompt["answer_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    )
                    + " "
                    + a
                    + "\n"
                )
            f.write(
                self.prompt["subquestion_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=len(state) + 1
                )
                + " "
                + action
                + "\n"
            )
            f.write(
                self.prompt["answer_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=len(state) + 1
                )
            )
            model_input = f.getvalue()

        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        result = ""
        for start1 in range(0, self.n_confidence, self.early_stop_base):
            stop1 = min(start1 + self.early_stop_base, self.n_confidence)

            for start in range(start1, stop1, self.batch_size):
                stop = min(start + self.batch_size, stop1)
                num = stop - start

                outputs = self.base_model.generate(
                    [model_input] * num,
                    hide_input=True,
                    do_sample=True,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    eos_token_id="\n",
                ).text
                for output in outputs:
                    result = output.strip()
                    answer = self.utils.retrieve_answer(result)
                    answer_dict[answer].append(result)

            # Early stop if confidence is high enough
            if len(answer_dict) == 0:  # no answer yet
                continue
            sorted_answer_dict = sorted(
                answer_dict.items(), key=lambda p: len(p[1]), reverse=True
            )
            max_len = len(sorted_answer_dict[0][1])
            if max_len / stop1 >= self.early_stop_threshold:
                if len(sorted_answer_dict) >= 2 and max_len == len(
                    sorted_answer_dict[1][1]
                ):
                    pass  # Tie with the second best answer
                else:
                    break

        if len(answer_dict) == 0:
            print("Warning: no answer found")
            confidence, answer = (
                0,
                result,
            )  # No reasonable answer found. Fall back to choose the last response
        else:
            sorted_answer_dict = sorted(
                answer_dict.items(), key=lambda p: len(p[1]), reverse=True
            )
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            max_len = len(max_answer_output_list)
            answer = max_answer_output_list[
                0
            ]  # Here we simply choose the first appearance of the answer
            confidence = max_len / sum(len(v) for v in answer_dict.values())

        state.append(SubResult(action, answer, confidence))
        aux = {"confidence": confidence}
        return state, aux

    def is_terminal(self, state: GSM8kState) -> bool:
        if len(state) > 0 and "Now we can answer" in state[-1].sub_question:
            return True
        else:
            return False


class ChatGSM8kWorldModel(GSM8kWorldModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.retrieve_answer = self.utils.retrieve_chat_answer

    def update_example(self, example, prompt) -> None:
        super(GSM8kWorldModel, self).update_example(example, prompt)
        self.prompt_examples = []
        for idx, example in enumerate(self.prompt["interactive_examples"]):
            self.prompt_examples.extend(example)
        self.n_shots = len(self.prompt["interactive_examples"])

    def get_step_model_input(self, state: GSM8kState, action: GSM8kAction):
        model_input = deepcopy(self.prompt_examples)
        model_input.append(self.prompt["instruction"])
        model_input.append({"role": "user", "content": self.example})

        reasoning_path, idx = self.prompt["answer_prefix"], 1
        for t in state:
            q, a = t[:2]
            reasoning_path += self.prompt["question_prefix"] % idx + q + "\n"
            reasoning_path += a + "\n\n"
            idx += 1
        reasoning_path += self.prompt["question_prefix"] % idx + action + "\n"
        model_input.append({"role": "assistant", "content": reasoning_path})
        return model_input

    def get_step_outputs(self, outputs, state, action, midlle_result: MiddleResult):
        prefix = self.prompt["useful_examples_prefix"] + self.example + "\n\n"
        idx = 1

        for t in state:
            q, a = t[:2]
            prefix += self.prompt["useful_question_prefix"] + q + "\n" + a + "\n\n"
            idx += 1
        score_inputs = []
        for output in outputs:
            score_inputs.append(
                prefix
                + self.prompt["useful_question_prefix"]
                + action
                + "\n"
                + output
                + "\n"
                + self.prompt["useful_prefix"]
            )
        return score_inputs, "fast_reward", outputs

    def revise_function(self, states):
        inputs = []

        for i in range(len(states) - 1):
            model_input = deepcopy(self.prompt_examples)
            model_input.append(self.prompt["instruction"])
            model_input.append({"role": "user", "content": self.example})

            reasoning_path, idx = self.prompt["answer_prefix"], 1
            for t in states[:i]:
                q, a = t[:2]
                reasoning_path += self.prompt["question_prefix"] % idx + q + "\n"
                reasoning_path += a + "\n\n"
                idx += 1
            inputs.append(
                model_input + [{"role": "assistant", "content": reasoning_path}]
            )
        return inputs, "revise"

    def get_step_outputs_finnal(
        self, outputs, state, action, score_logits, midlle_result: MiddleResult
    ):
        if len(outputs) == 0:
            confidence = -10
            state.append(SubResult(action, "", confidence))
            print("not found answer")
            return state, {"confidence": confidence}
        score_logits = np.exp(score_logits) / np.sum(
            np.exp(score_logits) + 1e-5, axis=-1, keepdims=1
        )
        confidences = score_logits[:, 0]
        confidence = confidences[np.argmax(confidences)]
        answer = outputs[np.argmax(confidences)]
        state.append(SubResult(action, answer, confidence))
        aux = {"confidence": confidence}
        return state, aux


class ChatCodeGSM8kWorldModel(ChatGSM8kWorldModel):
    def get_step_model_input(self, state: GSM8kState, action: GSM8kAction):
        model_input = deepcopy(self.prompt_examples)
        model_input.append(self.prompt["instruction"])
        model_input.append({"role": "user", "content": self.example})

        reasoning_path, idx = self.prompt["inital_variable"], 2
        for t in state:
            q, a = t[:2]
            reasoning_path += self.prompt["question_prefix"] % idx + q + "\n"
            reasoning_path += a + "\n\n"
            idx += 1
        reasoning_path += self.prompt["question_prefix"] % idx + action + "\n"
        model_input.append({"role": "assistant", "content": reasoning_path})
        return model_input

    def is_terminal(self, state: GSM8kState) -> bool:
        if len(state) > 0 and (
            "now we can answer" in state[-1].sub_question.lower()
            or "now we can answer" in state[-1].sub_answer.lower()
        ):
            return True
        else:
            return False

    def get_step_outputs(self, outputs, state, action, midlle_result: MiddleResult):
        prefix = (
            self.prompt["useful_examples_prefix"]
            + self.example
            + "\n\n"
            + self.prompt["inital_variable"]
        )
        idx = 1
        for t in state:
            q, a = t[:2]
            prefix += (
                self.prompt["useful_question_prefix"] % idx + q + "\n" + a + "\n\n"
            )
            idx += 1
        score_inputs = []
        for output in outputs:
            score_inputs.append(
                prefix
                + self.prompt["useful_question_prefix"]
                + action
                + "\n"
                + output
                + "\n"
                + self.prompt["useful_prefix"]
            )
        return score_inputs, "fast_reward", outputs


class CodeGSM8kWorldModel(GSM8kWorldModel):
    def update_example(self, example, prompt) -> None:
        super(GSM8kWorldModel, self).update_example(example, prompt)
        assert prompt is not None
        self.prompt = prompt

        self.prompt_examples = []
        for idx, example in enumerate(self.prompt["interactive_examples"]):
            for i in range(len(example)):
                if "{idx}" in example[i]["content"]:
                    example[i]["content"] = example[i]["content"].format(idx=idx + 1)
            self.prompt_examples.extend(example)
        self.n_shots = len(self.prompt["interactive_examples"])

    def get_step_model_input(self, state: GSM8kState, action: GSM8kAction):
        model_input = deepcopy(self.prompt_examples)
        model_input.append(
            {
                "role": "user",
                "content": self.prompt["question_prefix"].format(
                    idx=self.n_shots + 1, question=self.example
                ),
            }
        )
        model_input.append(self.prompt["get_var_prompt"])
        model_input.append(
            {
                "role": "assistant",
                "content": "```python"
                + self.prompt["known_variables_generate"]
                + "\n```",
            }
        )
        idx = 0
        for t in state:
            q, a = t[:2]
            model_input.append(
                {
                    "role": "user",
                    "content": self.prompt["subquestion_prefix"].format(
                        idx=self.n_shots + 1, sub_idx=idx + 1
                    ),
                }
            )
            model_input.append({"role": "assistant", "content": q})
            model_input.append(self.prompt["answer_prefix"])
            model_input.append({"role": "assistant", "content": a})
            idx += 1
        model_input.append(
            {
                "role": "user",
                "content": self.prompt["subquestion_prefix"].format(
                    idx=self.n_shots + 1, sub_idx=idx + 1
                ),
            }
        )
        model_input.append({"role": "assistant", "content": action})
        model_input.append(self.prompt["answer_prefix"])
        return model_input

    def get_step_outputs(self, outputs, state, action, midlle_result: MiddleResult):
        prefix = self.prompt["useful_examples_prefix"] + self.example + "\n"
        idx = 1
        for t in state:
            q, a = t[:2]
            prefix += self.prompt["useful_prompt"] % (q, a)
            idx += 1
        score_inputs = []
        for output in outputs:
            score_inputs.append(
                prefix
                + self.prompt["useful_prompt"] % (action, output)
                + self.prompt["useful_prefix"]
            )
        return score_inputs, "fast_reward", outputs

    def get_step_outputs_finnal(
        self, outputs, state, action, score_logits, midlle_result: MiddleResult
    ):
        score_logits = np.exp(score_logits) / np.sum(
            np.exp(score_logits) + 1e-5, axis=-1, keepdims=1
        )
        confidences = score_logits[:, 0]

        class SubResult(NamedTuple):
            sub_question: str
            sub_answer: str
            confidence: float
            code: str
            code_input: dict
            code_output: dict

        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        code_str, code_inputs, code_outputs = (
            midlle_result.exec_code,
            midlle_result.para_input,
            midlle_result.para_output,
        )

        result = ""
        for i, output in enumerate(outputs):
            if output is None:
                continue
            result = output.strip()
            answer = self.utils.retrieve_answer(result)
            code, code_input, code_output = code_str[i], code_inputs[i], code_outputs[i]
            answer_dict[answer].append(
                [result, code, code_input, code_output, confidences[i]]
            )
        sorted_answer_dict = sorted(
            answer_dict.items(), key=lambda p: len(p[1]), reverse=True
        )
        if len(answer_dict) == 0:
            print("Warning: no answer found")
            answer, confidence = "This question not found a code to solve it", 0
            code, code_input, code_output = "", "", ""
        else:
            sorted_answer_dict = sorted(
                answer_dict.items(), key=lambda p: len(p[1]), reverse=True
            )
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            answer, code, code_input, code_output, confidence = max_answer_output_list[
                0
            ]  # Here we simply choose the first appearance of the answer
        state.append(
            SubResult(action, answer, confidence, code, code_input, code_output)
        )
        aux = {"confidence": confidence}
        return state, aux


class Weak12KWorldModel(ChatGSM8kWorldModel):
    def is_terminal(self, state: GSM8kState) -> bool:
        if len(state) > 0 and "现在我们可以回答这个问题" in state[-1].sub_question:
            return True
        else:
            return False


class CodeWeak12KWorldModel(CodeGSM8kWorldModel):
    def get_step_outputs(self, outputs, state, action, midlle_result: MiddleResult):
        class SubResult(NamedTuple):
            sub_question: str
            sub_answer: str
            confidence: float
            code: str
            code_input: dict
            code_output: dict

        answer_dict = defaultdict(list)  # map from answer to list of thoughts
        code_str, code_inputs, code_outputs = (
            midlle_result.exec_code,
            midlle_result.para_input,
            midlle_result.para_output,
        )

        result = ""
        for i, output in enumerate(outputs):
            if output is None:
                continue
            result = output.strip()
            answer = self.utils.retrieve_answer(result)
            code, code_input, code_output = code_str[i], code_inputs[i], code_outputs[i]
            answer_dict[answer].append([result, code, code_input, code_output])
        sorted_answer_dict = sorted(
            answer_dict.items(), key=lambda p: len(p[1]), reverse=True
        )
        if len(answer_dict) == 0:
            print("Warning: no answer code found")
            answer, confidence = "This question not found a code to solve it", -1
            code, code_input, code_output = "", "", ""
        else:
            sorted_answer_dict = sorted(
                answer_dict.items(), key=lambda p: len(p[1]), reverse=True
            )
            max_answer = sorted_answer_dict[0]
            max_answer_output_list = max_answer[1]
            max_len = len(max_answer_output_list)
            answer, code, code_input, code_output = max_answer_output_list[
                0
            ]  # Here we simply choose the first appearance of the answer
            confidence = max_len / sum(len(v) for v in answer_dict.values())
        state.append(
            SubResult(action, answer, confidence, code, code_input, code_output)
        )
        aux = {"confidence": confidence}
        return state, aux

    def is_terminal(self, state: GSM8kState) -> bool:
        if len(state) > 0 and "现在我们可以回答这个问题" in state[-1].sub_question:
            return True
        else:
            return False


class ChatSVAMPWorldModel(ChatGSM8kWorldModel):
    def is_terminal(self, state: GSM8kState) -> bool:
        if len(state) > 0 and "now we can answer" in state[-1].sub_question.lower():
            return True
        else:
            return False


class DeepMctsWorldModel:
    def update_example(self, example, prompt) -> None:
        super(GSM8kWorldModel, self).update_example(example, prompt)

    def get_step_model_input(self, state: GSM8kState, action: GSM8kAction):
        model_input = []
        model_input.append(self.prompt["instruction"])
        model_input.append({"role": "user", "content": self.example})

        reasoning_path, idx = "", 1
        for t in state:
            q, a = t[:2]
            reasoning_path += self.prompt["question_prefix"] % (q) + "\n"
            reasoning_path += a + "\n\n"
            idx += 1
        reasoning_path += self.prompt["question_prefix"] % (action)
        model_input.append({"role": "assistant", "content": reasoning_path})
        return [model_input, action]

    def get_step_outputs(self, outputs, state, action, midlle_result: MiddleResult):
        if len(outputs) == 1 and self.n_iters == 1:
            state.append(SubResult(action, outputs[0], 1))
            return state, {"confidence": 1}
        temp = deepcopy(self.get_step_model_input(state, action)[0])
        prefix, reasoning_path = temp[:-1], temp[-1]["content"]
        score_inputs = []
        for output in outputs:
            score_inputs.append(
                prefix
                + [
                    {
                        "role": "assistant",
                        "content": reasoning_path + "\n" + output + "\n",
                    }
                ]
            )
        return score_inputs, "fast_reward", outputs

    def is_terminal(self, state: GSM8kState) -> bool:
        if state[-1].sub_question in self.prompt["summar_prompt"]:
            return True
        else:
            return False

    def get_step_outputs_finnal(
        self, outputs, state, action, score_logits, midlle_result: MiddleResult
    ):
        if action in self.prompt["summar_prompt"]:
            new_outputs = []
            for out in outputs:
                try:
                    if self.retrieve_answer(out) != None:
                        new_outputs.append(out)
                except:
                    pass
            outputs = new_outputs
        if len(outputs) == 0:
            confidence = -10
            state.append(SubResult(action, "", confidence))
            print("not found answer")
            return state, {"confidence": confidence}
        score_logits = np.exp(score_logits) / np.sum(
            np.exp(score_logits) + 1e-5, axis=-1, keepdims=1
        )
        confidences = score_logits[:, 0]
        confidence = confidences[np.argmax(confidences)]
        answer = outputs[np.argmax(confidences)]
        state.append(SubResult(action, answer, confidence))
        aux = {"confidence": confidence}
        return state, aux

    def revise_function(self, states):
        inputs = []
        action = self.prompt["summar_prompt"][0]
        for i in range(1, len(states)):
            model_input = []
            state = states[: i + 1]
            model_input = []
            model_input.append(deepcopy(self.prompt["instruction"]))
            model_input.append({"role": "user", "content": self.example})

            reasoning_path, idx = "", 1
            for t in state:
                q, a = t[:2]
                reasoning_path += self.prompt["question_prefix"] % (q) + "\n"
                reasoning_path += a + "\n\n"
                idx += 1
            reasoning_path += self.prompt["question_prefix"] % (action)
            model_input.append({"role": "assistant", "content": reasoning_path})
            inputs.append(model_input)
        return inputs, "revise"


class Weak12KDeepMctsWorldModel(DeepMctsWorldModel, Weak12KWorldModel):
    pass


class GSM8kDeepMctsWorldModel(DeepMctsWorldModel, ChatGSM8kWorldModel):
    pass
