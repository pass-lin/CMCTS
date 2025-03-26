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
from reasoners.benchmark import MathEvaluator
from utils import gsm8k_utils
from tqdm import tqdm
from prompt import school_prompt as prompt
from latex2sympy2 import latex2sympy

model = ChatCOTModel(
    model_name,
    prompt=prompt,
    stop=None,
    top_p=0.8,
    top_k=32,
    max_tokens=4096,
    max_model_len=4096,
    temperature=1.0,
)


n_iters: int = 9


search_algo_params = {}
evaluator = MathEvaluator(
    output_extractor=gsm8k_utils.retrieve_chat_middleschool_answer,
    answer_extractor=lambda x: [
        gsm8k_utils.fullwidth_to_halfwidth(x["answer"])
        .replace(" ", "")
        .replace("%", "/100"),
        x["choice_answer"].replace(" ", ""),
    ],
    filename="data/cn_middle_school/test.jsonl",
    init_prompt=prompt,
    sample_prompt_type="rap",
)
dataset = list(evaluator.full_dataset)[:]

mcts_list = []
reasoners = []
correct_count = 0
examples = []


def eval_acc(output, answer):
    pred, flag = output
    answer_str, choice = answer
    try:
        if flag == "sympy":
            answer_str = latex2sympy(answer_str)
            return (answer_str == pred) or (answer_str.simplify() == pred.simplify())
        elif flag == "option":
            return choice == pred
        else:
            return pred == answer_str
    except:
        return False


def eval_output(out1, out2):
    pred1, flag1 = out1
    pred2, flag2 = out2
    try:
        if flag2 != flag1:
            return False
        if flag1 == "sympy":
            return pred1.simplify() == pred2.simplify()
        else:
            return pred1 == pred2
    except:
        return False


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
        answer = evaluator.answer_extractor(dataset[i])
        outputs = []
        outputs_set = []
        counts = []
        # 提取每次迭代中的输出
        paths = []
        for t in model_outputs[i][:iter]:
            output = evaluator.output_extractor(t["content"])
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

    print("maj@%d:%f" % (iter, count_correct / len(model_inputs)))
