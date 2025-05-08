# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:50:09 2024

@author: Administrator
"""

import numpy as np
import os
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_name = "Qwen/Qwen2.5-7B-Instruct"

from LLM import *
from reasoners.benchmark import Game24Evaluator
from utils import game24_utils
from tqdm import tqdm
from prompt import game24_prompt as prompt
import sympy

evaluator = Game24Evaluator(
    output_extractor=game24_utils.retrieve_chat_game24_answer,
    filename="data/game_24.csv",
    init_prompt=prompt,
    sample_prompt_type="rap",
)
dataset = examples = list(evaluator.full_dataset)[:]

model = ChatCOTModel(
    model_name,
    prompt=prompt,
    stop=prompt["stop"],
    select_tokens=[" 是", " 否"],
    top_p=0.7,
    top_k=50,
    max_tokens=4096,
    max_model_len=4096,
    temperature=1.0,
    gpu_memory_utilization=0.7,
)


n_iters: int = 9





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
    nums = dataset[i].split(" ")
    question =deepcopy(prompt["user_prefix"])
    for num in set(nums):
        question += "\n最多使用%d次%d"%(nums.count(num),int(num))
    model_input.append({"role": "user", "content": question})
    model_input.append({"role": "assistant", "content": "让我们一步步思考来解决这个问题:\n"})
    model_inputs.append(model_input)
outputs = model.chat_generate(model_inputs, n=n_iters)
model_outputs = []
for i in range(len(model_inputs)):
    model_outputs.append(outputs[i * n_iters : (i + 1) * n_iters])




for iter in range(1, n_iters + 1):
    count_correct = 0
    total = 0
    for i in range(len(model_inputs)):
        try:
            total += 1

            # 提取每次迭代中的输出
            outputs = []
            for t in model_outputs[i][:iter]:
                try:
                    output =evaluator.output_extractor(t["content"])
                    if output == "None" or output == None:
                        continue
                    outputs.append(output)
                except:
                    pass


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
            candidate = dataset[i].split(" ")
            count_flag = game24_utils.eval_acc(count_output, candidate)
            count_correct += count_flag
        except:
            pass
    print("maj@%d:%f" % (iter, count_correct / total))
