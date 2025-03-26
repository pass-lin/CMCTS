# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:50:09 2024

@author: Administrator
"""

import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_name = "Qwen/Qwen2.5-14B-Instruct"

from LLM import *
from reasoners.benchmark import Weak12kEvaluator
from utils import gsm8k_utils
from tqdm import tqdm
from prompt import weak12k_prompt as prompt


model = ChatCOTModel(
    model_name,
    prompt=prompt,
    stop=None,
    top_p=0.7,
    top_k=50,
    max_tokens=4096,
    max_model_len=4096,
    temperature=1.05,
)


n_iters: int = 9


evaluator = Weak12kEvaluator(
    output_extractor=gsm8k_utils.retrieve_chat_weak12k_answer,
    answer_extractor=lambda x: x["answer"],
    filename="data/Weak12K/weekly12k_test_clean.json",
    init_prompt=prompt,
    sample_prompt_type="rap",
)
dataset = examples = list(evaluator.full_dataset)[:]


model_inputs = []
for i, example in enumerate(
    tqdm(
        dataset[:],
        total=len(dataset),
        desc=evaluator._dataset_name,
    )
):
    model_input = []
    model_input.append(prompt["instruction"])
    model_input.append({"role": "user", "content": example["question"]})
    model_input.append({"role": "assistant", "content": "Let's think step by step:\n"})
    model_inputs.append(model_input)
outputs = model.chat_generate(model_inputs, n=n_iters)
model_outputs = []
for i in range(len(model_inputs)):
    model_outputs.append(outputs[i * n_iters : (i + 1) * n_iters])


def eval_acc(output, answer):
    try:
        answer = eval(str(answer))
        output = abs(
            eval(str(output))
        )  # 这个数据集没有负数的答案，但是题目可能让题目回答负数，比如水底下50米表达是-50，但答案是50
        flags = []
        for t in [
            output / 100,
            output,
            output * 100,
        ]:  # 考虑百分数的不同回答情况，当然，baseline也是有做出相应修正的
            # 这里的精度取1e-2，因为大模型经常会四舍五入到小数点三位，baseline也会做出对应的修正
            flags.append(abs(t - answer) <= 1e-2)
        return any(flags)
    except:
        pass
    return False


for iter in range(1, n_iters + 1):
    count_correct = 0
    total = 0
    for i in range(len(model_inputs)):
        try:
            if (
                examples[i]["question"].count("?") > 2
                or "提出" in examples[i]["question"]
            ):
                # 会存在一些奇葩题目在这个数据集里，要做修正
                # 还有比如”你再提出一个问题巴拉巴拉“这种奇葩问题用规则过滤一下
                # 之所以在这里过滤而不在选择数据那里过滤，是因为在那里过滤性能会莫名其妙下降，我也不知为什么
                continue
            total += 1
            # 提取当前示例的答案
            answer = examples[i]["answer"]
            outputs = []
            if int(answer) == float(answer):
                answer = str(int(answer))  # 如果答案是整数，转换为整数字符串
            else:
                answer = str(answer)  # 否则保持浮点数字符串

            # 提取每次迭代中的输出
            paths = []
            for t in model_outputs[i][:iter]:
                try:
                    output = str(evaluator.output_extractor(t["content"]))
                    output = eval(output)
                    assert output != None  # 尝试评估输出是否为有效表达式
                except:
                    try:
                        # 如果无效，尝试从子问题中提取输出并评估
                        output = str(
                            evaluator.output_extractor(
                                "\n\n".join(
                                    [
                                        a.sub_question + "\n" + a.sub_answer
                                        for a in t[-1].state[:-1]
                                    ]
                                )
                                + t[-1].state[-1].sub_question
                            )
                        )
                        output = eval(output)
                    except:
                        continue  # 如果仍然无效，跳过该路径
                if output == "None" or output == None:
                    continue
                if int(output) == float(output):
                    output = int(output)  # 如果答案是整数，转换为整数字符串
                else:
                    output = round(output, 3)
                output = str(output)
                outputs.append(str(output))
                paths.append(t)

            # 如果没有输出且搜索空间为空，则重新初始化MCTS实例
            # if len(outputs) == 0 and mcts_list[i].is_no_search_space():
            #    mcts_list[i].initial(mcts_list[i].world_model, mcts_list[i].search_config)
            if len(outputs) == 0:
                continue
            # 获取唯一的输出集合
            outputs_set = list(set(outputs))

            # 找到出现次数最多的输出
            counts = [outputs.count(a) for a in outputs_set]
            count_output = outputs_set[np.argmax(counts)]

            # 更新计数策略的正确率
            count_flag = eval_acc(count_output, answer)
            count_correct += count_flag
        except:
            pass
    print("maj@%d:%f" % (iter, count_correct / total))
