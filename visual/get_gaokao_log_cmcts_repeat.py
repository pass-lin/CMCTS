import numpy as np
import os
import time
#生成可视化需要的样例，需要注意的是我们将topp拉满，避免因为超参数的原因导致状态多样性上限受限
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_name = "Qwen/Qwen2.5-7B-Instruct"
reward_model_name = "Qwen/Qwen2.5-Math-PRM-7B"


from LLM import *
from reasoners.benchmark import MathEvaluator
from reasoners import Reasoner
from reasoners.algorithm import MCTS, MiddleResult
from world_model import GSM8kDeepMctsWorldModel
from search_config import DeepGSM8kConfig
from utils import gsm8k_utils
from copy import deepcopy
from tqdm import tqdm
import string
from math import isclose
from prompt import gaokaoen_prompt as prompt
from sympy import simplify, N


partial_order = [False] * 5
native_rewards_mode = False  # 是否使用自身作为PRM
# wo paln+reflect用3
model = DeepMCTSModel4(
    model_name,
    prompt=prompt,
    reward_model_name=reward_model_name,
    stop=None,
    top_p=1,
    top_k=512,
    max_tokens=512,
    max_model_len=4096,
    temperature=1,
    gpu_memory_utilization=0.9 if native_rewards_mode else 0.5,
    reward_model_gpu_memory_utilization=1,
    native_rewards_mode=native_rewards_mode,
)

n_confidence: int = 3
depth_limit: int = 7
cum_reward = np.sum
n_iters: int = 7
max_sample_num = 10

search_algo_params = {}
evaluator = MathEvaluator(
    output_extractor=gsm8k_utils.retrieve_chat_gaokaoen_answer,
    filename="data/gaokao2023en/test.jsonl",
    init_prompt=prompt,
    sample_prompt_type="rap",
)

dataset = list(evaluator.full_dataset)[:16]
historys = [[] for i in range(len(dataset))]
for _ in range(max_sample_num):
    mcts_list = []
    reasoners = []
    correct_count = 0
    examples = []
    for i, example in enumerate(
        tqdm(
            dataset,
            total=len(dataset),
            desc=evaluator._dataset_name,
        )
    ):
        try:
            world_model = GSM8kDeepMctsWorldModel(
                n_confidence=n_confidence,
                n_iters=n_iters,
                retrieve_answer=evaluator.output_extractor,
            )
            config = DeepGSM8kConfig(
                force_terminating_on_depth_limit=True,
                partial_order=partial_order,
                depth_limit=depth_limit,
            )
            search_algo = MCTS(
                cum_reward=cum_reward,
                rule_action=True,
                depth_limit=depth_limit,
                uct_with_fast_reward=False,
            )
            mcts_list.append(search_algo)
            reasoner = Reasoner(
                world_model=world_model, search_config=config, search_algo=search_algo
            )
            reasoner.update(example["question"], prompt)

            reasoner.search_algo.initial(reasoner.world_model, reasoner.search_config)
            reasoners.append(reasoner)
            examples.append(example)
        except:
            mcts_list.pop(-1)
    # 开始推理

    print(len(mcts_list))
    middle_results = [MiddleResult() for i in range(len(mcts_list))]
    iterates = [
        mcts_list[i].parallel_iterate(TempResult=middle_results[i])
        for i in range(len(mcts_list))
    ]
    start = time.time()
    for iter in range(n_iters):
        print("iter num " + str(iter))
        flags = np.ones(len(mcts_list), "bool")
        steps = 1
        result_path = []

        while np.any(flags):
            model_inputs, states = [], []
            for i in range(len(mcts_list)):
                if flags[i]:
                    model_input, state = next(iterates[i])

                    if state == "Search_End":
                        path = model_input[:]
                        mcts_list[i].get_cum_reward(path)
                        mcts_list[i].trace_in_each_iter.append(deepcopy(path))
                        flags[i] = False
                    elif state == "fast_reward":
                        middle_results[i].reward_prompt = model_input
                    elif state == "step":
                        middle_results[i].step_prompt = model_input
                    elif state == "get_action":
                        middle_results[i].action_prompt = model_input
                else:
                    model_input, state = None, None
                model_inputs.append(model_input)
                states.append(state)
            print(len(states), sum(flags))
            print(steps)
            print(set(states))
            # if steps==2:raise(1)
            model.generate(model_inputs, flags, states, middle_results)
            for i, state in enumerate(states):
                if state == "step":
                    steps_output = []
                    for t in middle_results[i].step_outputs:
                        if len(model.tokenizer.encode(t)) < model.max_tokens * 2:
                            steps_output.append(t)
                    middle_results[i].step_outputs = steps_output

            steps += 1
    for i in range(len(mcts_list)):
        for t in mcts_list[i].trace_in_each_iter:
            historys[i].append(
                [[state.sub_question, state.sub_answer] for state in t[-1].state]
            )


import json

sequences = []
for i in range(len(mcts_list)):
    sequences.append({"question": dataset[i], "level_list": historys[i]})
with open("history_cmcts_repeat.json", "w") as f:
    json.dump(sequences, f, indent=4)
