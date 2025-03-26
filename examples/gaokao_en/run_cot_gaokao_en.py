# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:50:09 2024

@author: Administrator
"""

import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
from LLM import *
from reasoners.benchmark import MathEvaluator
from utils import gsm8k_utils
from tqdm import tqdm
import string
from math import isclose
from prompt import math_prompt as prompt
from sympy import simplify, N

model = ChatCOTModel(
    model_name,
    prompt=prompt,
    stop=None,
    gpu_memory_utilization=0.95,
    top_p=0.8,
    top_k=64,
    max_tokens=4096,
    max_model_len=4096,
    temperature=1.0,
)
n_iters: int = 7


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
                if not isclose(float(N(a[i])), float(N(b[i])), rel_tol=1e-4):
                    return False
            return True

        if type(a) == list:
            if any([caculate(a, b), caculate(a, b[::-1])]):
                return True
        elif isclose(float(N(a)), float(N(b)), rel_tol=1e-4):
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
    output_extractor=gsm8k_utils.retrieve_chat_gaokaoen_answer,
    filename="data/gaokao2023en/test.jsonl",
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

for iter in range(1, n_iters + 1):
    count_correct = 0
    for i in range(len(model_inputs)):
        try:
            options = None
            if examples[i]["answer"][:3] == "$x=":
                examples[i]["answer"] = examples[i]["answer"][3:]
            if examples[i]["answer"] in string.ascii_uppercase:
                option_label = ""
                choice_string = ""
                options = []
                for temp_str in examples[i]["question"].split(":")[-1].split("("):
                    if len(temp_str) < 2:
                        continue
                    elif temp_str[1] == ")":
                        if option_label != "":
                            options.append(
                                [
                                    option_label,
                                    choice_string.replace("$", "").replace(" ", ""),
                                ]
                            )
                        option_label = temp_str[0]
                        choice_string = temp_str[2:]
                    else:
                        choice_string += temp_str
                options.append(
                    [option_label, choice_string.replace("$", "").replace(" ", "")]
                )

            answer = evaluator.output_extractor(examples[i]["answer"], False, options)
            outputs = []
            outputs_set = []
            counts = []
            # 提取每次迭代中的输出
            paths = []
            for t in model_outputs[i][:iter]:
                output = evaluator.output_extractor(t["content"], options=options)
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

            if len(outputs) == 0:
                continue

            counts = [outputs.count(a) for a in outputs_set]
            count_output = outputs_set[np.argmax(counts)]

            # 更新计数策略的正确率
            count_flag = eval_acc(count_output, answer)
            count_correct += count_flag
        except:
            pass

    print("maj@%d:%f" % (iter, count_correct / len(model_inputs)))
