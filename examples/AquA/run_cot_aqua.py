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
from reasoners.benchmark import AQuAEvaluator
from utils import aqua_utils
from tqdm import tqdm
from prompt import aqua_prompt as prompt

model = ChatCOTModel(
    model_name,
    prompt=prompt,
    stop=None,
    top_p=0.7,
    top_k=32,
    max_tokens=2048,
    max_model_len=2048,
    temperature=1,
    gpu_memory_utilization=0.9,
)

n_iters: int = 9


search_algo_params = {}
evaluator = evaluator = AQuAEvaluator(
    output_extractor=aqua_utils.retrieve_chat_aqua_answer,
    answer_extractor=lambda x: aqua_utils.retrieve_answer_from_dataset(x["answer"]),
    init_prompt=prompt,
    sample_prompt_type="rap",
    dataset_path="data/AQuA/test.json",
)
dataset = list(evaluator.full_dataset)[:]
mcts_list = []
reasoners = []
correct_count = 0
examples = []


def eval_acc(output, answer):
    try:
        if output == answer:
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
        answer = evaluator.answer_extractor(dataset[i])
        outputs = []
        for t in model_outputs[i][:iter]:
            output = str(
                evaluator.output_extractor(t["content"], dataset[i]["options"])
            )
            outputs.append(output)
        if len(outputs) == 0:
            continue
        outputs_set = list(set(outputs))

        # 找到出现次数最多的输出
        counts = [outputs.count(a) for a in outputs_set]
        count_output = outputs_set[np.argmax(counts)]

        # 更新计数策略的正确率
        count_flag = eval_acc(count_output, answer)
        count_correct += count_flag

    print("maj@%d:%f" % (iter, count_correct / len(model_inputs)))
