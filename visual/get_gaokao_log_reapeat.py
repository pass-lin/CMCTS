import numpy as np
import os
import time
#生成可视化需要的样例，需要注意的是我们将topp拉满，避免因为超参数的原因导致状态多样性上限受限
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_name = "Qwen/Qwen2.5-7B-Instruct"
reward_model_name = "Qwen/Qwen2.5-Math-PRM-7B"


from LLM import *
from reasoners.benchmark import MathEvaluator
from reasoners import Reasoner
from reasoners.algorithm import MCTS, MiddleResult
from world_model import ChatGSM8kWorldModel as GSM8kWorldModel
from search_config import ChatGSM8kConfig as GSM8kConfig
from utils import gsm8k_utils
from copy import deepcopy
from tqdm import tqdm
import string
from math import isclose
from sympy import simplify, N
from rap_prompt import mathprompt as prompt
from LLM import ChatCOTModel
from rap_prompt import gsm8k_usefulprompt as useful_prompt

model = PRMChatCOTModel(
    model_name,
    prompt=prompt,
    reward_model_name=reward_model_name,
    stop=prompt["stop"],
    top_p=1,
    top_k=512,
    max_tokens=512,
    max_model_len=4096,
    temperature=1,
    gpu_memory_utilization=0.6,
    reward_model_gpu_memory_utilization=0.8,
)
n_confidence: int = 3
depth_limit: int = 7
cum_reward = np.sum
n_iters: int = 7
n_action: int = 8
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
            world_model = GSM8kWorldModel(n_confidence=n_confidence)
            config = GSM8kConfig(
                useful_prompt=useful_prompt,
                n_actions=n_action,
                force_terminating_on_depth_limit=True,
                depth_limit=depth_limit,
            )
            search_algo = MCTS(
                cum_reward=cum_reward,
                output_strategy="max_reward",
                depth_limit=depth_limit,
            )
            mcts_list.append(search_algo)
            reasoner = Reasoner(
                world_model=world_model, search_config=config, search_algo=search_algo
            )

            config.force_overall_prompt_on_overall_question = False
            config.force_overall_question_on_overall_prompt = False
            reasoner.update(example["question"], prompt)
            reasoner.search_algo.initial(reasoner.world_model, reasoner.search_config)
            reasoners.append(reasoner)
            examples.append(example)
        except:
            mcts_list.pop(-1)
    # 开始推理
    start = time.time()
    print(len(mcts_list))
    middle_results = [MiddleResult() for i in range(len(mcts_list))]
    iterates = [
        mcts_list[i].parallel_iterate(TempResult=middle_results[i])
        for i in range(len(mcts_list))
    ]
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
                        path = model_input
                        mcts_list[i].get_cum_reward(path)
                        mcts_list[i].trace_in_each_iter.append(deepcopy(path))
                        flags[i] = False
                    elif state == "fast_reward":
                        middle_results[i].reward_prompt = model_input
                    elif state == "step":
                        for i in range(len(model_input)):
                            model_input[i][-1]["content"] = model_input[i][-1][
                                "content"
                            ].replace(
                                prompt["overall_question_prefix"]
                                + prompt["overall_question_prefix"],
                                prompt["overall_question_prefix"],
                            )
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

            model.generate(model_inputs, flags, states, middle_results, n_action)
            steps += 1
            for i, state in enumerate(states):
                if state == "get_action":
                    for j in range(len(middle_results[i].action_outputs)):
                        if (
                            "now we can" in middle_results[i].action_outputs[j].lower()
                            or "**\n" == middle_results[i].action_outputs[j]
                        ):
                            middle_results[i].action_outputs[j] = (
                                prompt["overall_question_prefix"]
                                + dataset[i]["question"]
                                + prompt["question_postfix"]
                                + "\nwe need to write the final answer in \\boxed{}."
                            )
    for i in range(len(mcts_list)):
        for t in mcts_list[i].trace_in_each_iter:
            historys[i].append(
                [[state.sub_question, state.sub_answer] for state in t[-1].state]
            )
import json

sequences = []
for i in range(len(mcts_list)):
    sequences.append({"question": dataset[i], "level_list": historys[i]})
with open("history_native_repeat.json", "w") as f:
    json.dump(sequences, f, indent=4)
