# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:50:09 2024

@author: Administrator
"""

import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_name = "Qwen/Qwen2.5-7B-Instruct"

from LLM import *
from reasoners.benchmark import GSM8KEvaluator
from utils import gsm8k_utils
from tqdm import tqdm
from prompt import math_prompt as prompt


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

n_iters: int = 7


search_algo_params = {}
evaluator = GSM8KEvaluator(
    output_extractor=gsm8k_utils.retrieve_chat_answer,
    answer_extractor=gsm8k_utils.retrieve_answer_from_dataset,
    init_prompt=prompt,
    sample_prompt_type="rap",
)
dataset = examples = list(evaluator.full_dataset)[:]
mcts_list = []
reasoners = []
correct_count = 0
examples = []


def eval_acc(output, answer):
    try:
        if abs(float(answer) - float(output)) <= 1e-4:
            return 1
    except:
        pass
    return 0


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
        # 提取当前示例的答案
        answer = evaluator.answer_extractor(dataset[i]["answer"])
        outputs = []
        if int(answer) == float(answer):
            answer = str(int(answer))  # 如果答案是整数，转换为整数字符串
        else:
            answer = str(answer)  # 否则保持浮点数字符串

        # 提取每次迭代中的输出
        paths = []
        for t in model_outputs[i][:iter]:
            output = str(evaluator.output_extractor(t["content"]))
            try:
                output = eval(output)  # 尝试评估输出是否为有效表达式
            except:
                continue  # 如果仍然无效，跳过该路径
            if int(output) == float(output):
                output = int(output)  # 如果答案是整数，转换为整数字符串
            output = str(output)
            outputs.append(output)
            paths.append(t)

        # 如果没有输出且搜索空间为空，则重新初始化MCTS实例
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
    print("maj@%d:%f" % (iter, count_correct / len(model_inputs)))
