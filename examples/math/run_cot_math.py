# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:50:09 2024

@author: Administrator
"""

import numpy as np
import os
from prompt import math_prompt as prompt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_name = "Qwen/Qwen2.5-14B-Instruct"

from LLM import *
from reasoners.benchmark import MathEvaluator
from utils import gsm8k_utils
from utils.gsm8k_utils import math_answer_clean
from tqdm import tqdm
from math import isclose

from sympy import simplify, N

model = ChatCOTModel(
    model_name,
    prompt=prompt,
    stop=prompt["stop"],
    top_p=0.5,
    top_k=32,
    max_tokens=4096,
    max_model_len=4096,
    temperature=1,
)

n_confidence: int = 3
depth_limit: int = 7
cum_reward = np.sum
n_iters: int = 9
partial_order = True


def eval_acc(a, b):
    if type(a) == list or type(b) == list:
        a = list(a)
        b = list(b)
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:

        def caculate(a, b):
            if len(a) != len(b):
                return False
            for i in range(len(a)):
                if not isclose(float(N(a[i])), float(N(b[i]))):
                    return False
            return True

        if type(a) == list:
            if any([caculate(a, b), caculate(a, b[::-1])]):
                return True
        elif isclose(float(N(a)), float(N(b))):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


evaluator = MathEvaluator(
    output_extractor=gsm8k_utils.retrieve_chat_math_answer,
    filename="data/math_500/test.jsonl",
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
from tqdm import trange

for iter in range(1, n_iters + 1):
    count_correct = 0
    for i in trange(len(model_inputs)):
        # 提取当前示例的答案
        try:
            answer = math_answer_clean(examples[i]["answer"])
            if "\\text" in answer and answer[:5] != "\\text":
                answer = answer.split("\\text")[0]
            elif "\\text" in answer:
                answer = (
                    answer.split("\\text{")[1][:-1].replace(")", "").replace("(", "")
                )
            answer = evaluator.output_extractor(answer, False)
            outputs = []
            outputs_set = []
            counts = []
            # 提取每次迭代中的输出
            paths = []
            for t in model_outputs[i][:iter]:
                try:
                    output = evaluator.output_extractor(t["content"])
                except:
                    continue
                if output == None:
                    continue
                outputs.append(output)
                paths.append(t)
                if output not in outputs_set:
                    outputs_set.append(output)
                    counts.append(1)
                else:
                    index = outputs_set.index(output)
                    counts[index] += 1

            # 如果没有输出且搜索空间为空，则重新初始化MCTS实例
            # if len(outputs) == 0 and mcts_list[i].is_no_search_space():
            #    mcts_list[i].initial(mcts_list[i].world_model, mcts_list[i].search_config)
            if len(outputs) == 0:
                continue
            # 获取唯一的输出集合

            count_output = outputs_set[np.argmax(counts)]

            # 更新计数策略的正确率
            count_flag = eval_acc(count_output, answer)
            count_correct += count_flag
        except:
            pass
    print("maj@%d:%f" % (iter, count_correct / len(model_inputs)))
