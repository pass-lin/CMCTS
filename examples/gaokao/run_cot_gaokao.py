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
from reasoners.benchmark import GaokaoEvaluator
from utils import gsm8k_utils
from tqdm import tqdm
from prompt import gaokao_prompt as prompt

# 主实验用3，但是在做没有prm的消融的时候要换到4
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

evaluator = GaokaoEvaluator(
    output_extractor=gsm8k_utils.retrieve_chat_gaokao_answer,
    filename="data/gaokao_math_qa/test.jsonl",
    init_prompt=prompt,
    sample_prompt_type="rap",
)
dataset = list(evaluator.full_dataset)[:]
n_iters: int = 7


def eval_acc(output, answer):
    return output == answer


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
        answer = dataset[i]["label"].replace(" ", "")
        outputs = []
        outputs_set = []
        counts = []
        # 提取每次迭代中的输出
        paths = []
        for t in model_outputs[i][:iter]:
            output = evaluator.output_extractor(t["content"], dataset[i]["options"])
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
