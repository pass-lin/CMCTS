# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:50:09 2024

@author: Administrator
"""

import numpy as np
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
model_name = "Qwen/Qwen2.5-7B-Instruct"
reward_model_name = "Qwen/Qwen2.5-Math-PRM-7B"

from LLM import PRMChatCOTModel
from rap_prompt import aquaprompt as prompt
from rap_prompt import gsm8k_usefulprompt as useful_prompt


prompt["useful_instrution"] = prompt["useful_instrution_PRM"]
prompt["useful_prefix"] = prompt["useful_prefix_PRM"]

useful_prompt["useful_prefix"] = useful_prompt["useful_prefix_PRM"]
useful_prompt["input"] = useful_prompt["input_PRM"]

model = PRMChatCOTModel(
    model_name,
    prompt=prompt,
    reward_model_name=reward_model_name,
    stop=prompt["stop"],
    top_p=0.5,
    top_k=32,
    max_tokens=512,
    max_model_len=4096,
    temperature=1,
    gpu_memory_utilization=0.5,
    reward_model_gpu_memory_utilization=0.8,
)

n_action: int = 8
n_confidence: int = 3
depth_limit: int = 8
cum_reward = np.sum
n_iters: int = 9

from LLM import *
from reasoners.benchmark import AQuAEvaluator
from reasoners import Reasoner
from reasoners.algorithm import MCTS, MiddleResult
from world_model import ChatGSM8kWorldModel as GSM8kWorldModel
from search_config import ChatGSM8kConfig as GSM8kConfig
from utils import aqua_utils
from copy import deepcopy
from tqdm import tqdm


def eval_acc(output, answer):
    try:
        if output == answer:
            return 1
    except:
        pass
    return 0


search_algo_params = {}
evaluator = AQuAEvaluator(
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
for i, example in enumerate(
    tqdm(
        dataset,
        total=len(dataset),
        desc=evaluator._dataset_name,
        disable=evaluator.disable_tqdm,
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
            cum_reward=cum_reward, output_strategy="max_reward", depth_limit=depth_limit
        )
        mcts_list.append(search_algo)
        reasoner = Reasoner(
            world_model=world_model, search_config=config, search_algo=search_algo
        )
        reasoner.update(evaluator.input_processor(example), prompt)

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
                            + "\nwe need to choose the option from “A”, “B”, “C”, “D”, “E” write in \\boxed{}."
                        )
        steps += 1

    def eval_acc(output, answer):
        return output == answer

    def scores(
        outputs_set,
        outputs,
        trace_in_each_iter,
        answer,
        mean_mean_reward_correct,
        mean_sum_reward_correct,
        scores_function=np.mean,
        reward2score=True,
    ):
        """
        计算不同输出的平均分数和总分数，并更新给定的正确率统计。

        参数:
        - outputs_set: 所有可能的输出集合
        - outputs: 实际产生的输出序列
        - trace_in_each_iter: 每次迭代中产生的轨迹（包含奖励信息）
        - answer: 正确答案
        - mean_mean_reward_correct: 平均分数方法的累计正确率
        - mean_sum_reward_correct: 总分数方法的累计正确率
        - scores_function: 计算分数的方法，默认为np.mean

        返回:
        - mean_mean_reward_correct: 更新后的平均分数方法的累计正确率
        - mean_sum_reward_correct: 更新后的总分数方法的累计正确率
        """
        # 初始化分数列表，对应每个可能的输出
        scores = [[] for _ in range(len(outputs_set))]

        # 遍历实际产生的输出，计算每个输出的分数
        for j, out in enumerate(outputs):
            # 使用给定的分数计算方法计算分数，并添加到对应的输出分数列表中
            if reward2score:
                scores[outputs_set.index(out)].append(
                    scores_function([a.reward for a in trace_in_each_iter[j]])
                )
            else:
                scores[outputs_set.index(out)].append(
                    scores_function(
                        [a.confidence for a in trace_in_each_iter[j][-1].state]
                    )
                )
        # 计算每个输出的平均分数
        mean_scores = [np.mean(a) for a in scores]
        # 计算每个输出的总分数
        sum_scores = [np.sum(a) for a in scores]

        # 找到总分数最高的输出
        sum_reward_output = outputs_set[np.argmax(sum_scores)]
        # 找到平均分数最高的输出
        mean_reward_output = outputs_set[np.argmax(mean_scores)]

        # 更新平均分数方法的累计正确率
        mean_mean_reward_correct += eval_acc(mean_reward_output, answer)
        # 更新总分数方法的累计正确率
        mean_sum_reward_correct += eval_acc(sum_reward_output, answer)

        return mean_mean_reward_correct, mean_sum_reward_correct

    # 初始化各种正确率统计变量
    mean_mean_reward_correct = 0  # 平均奖励的平均值策略正确率
    mean_sum_reward_correct = 0  # 平均奖励的总和策略正确率
    sum_mean_reward_correct = 0  # 总奖励的平均值策略正确率
    sum_sum_reward_correct = 0  # 总奖励的总和策略正确率

    mean_mean_confidence_correct = 0  # 平均置信度的平均值策略正确率
    mean_sum_confidence_correct = 0  # 平均置信度的总和策略正确率
    sum_mean_confidence_correct = 0  # 总置信度的平均值策略正确率
    sum_sum_confidence_correct = 0  # 总置信度的总和策略正确率

    max_reward_path_corrct = 0  # 最高平均奖励路径策略正确率
    max_confidence_path_corrct = 0  # 最高平均置信度路径策略正确率

    max_terminal_reward_correct = 0  # 最后一个状态最高奖励策略正确率
    max_terminal_confidence_correct = 0  # 最后一个状态最高置信度策略正确率

    count_correct = 0  # 计数策略正确率
    no_answer_indexs = []  # 没有找到答案的索引列表

    # 遍历每个MCTS实例进行评估
    for i in range(len(mcts_list)):
        try:
            # 提取当前示例的答案
            answer = evaluator.answer_extractor(examples[i])
            outputs = []

            # 提取每次迭代中的输出
            paths = []
            for t in mcts_list[i].trace_in_each_iter:
                try:
                    if "\\boxed" in t[-1].state[-1].sub_answer.replace("\\boxed{}", ""):
                        output = evaluator.output_extractor(
                            t[-1].state, examples[i]["options"]
                        )
                    else:
                        output = evaluator.output_extractor(
                            "".join(t.sub_answer for t in t[-1].state),
                            examples[i]["options"],
                        )
                except:
                    continue
                outputs.append(output)
                paths.append(t)

            # 如果没有输出且搜索空间为空，则重新初始化MCTS实例
            if len(outputs) == 0 and mcts_list[i].is_no_search_space():
                mcts_list[i].initial(
                    mcts_list[i].world_model, mcts_list[i].search_config
                )

            # 获取唯一的输出集合
            outputs_set = list(set(outputs))

            # 找到出现次数最多的输出
            counts = [outputs.count(a) for a in outputs_set]
            count_output = outputs_set[np.argmax(counts)]

            # 更新计数策略的正确率
            count_flag = eval_acc(count_output, answer)
            count_correct += count_flag

            # 如果答案不在输出集中，记录索引
            if answer not in outputs_set:
                no_answer_indexs.append(i)

            if not count_flag:
                print(i, answer, outputs)
                print(examples[i]["question"])
                print(paths[outputs.index(count_output)][-1].state[-1].sub_answer)
                print("*" * 20)

            # 如果存在多个相同次数的输出或只有一个输出，以count为准
            if len(counts) <= 1 or np.sort(counts)[-1] != np.sort(counts)[-2]:
                mean_mean_reward_correct += count_flag
                mean_sum_reward_correct += count_flag
                sum_mean_reward_correct += count_flag
                sum_sum_reward_correct += count_flag

                mean_mean_confidence_correct += count_flag
                mean_sum_confidence_correct += count_flag
                sum_mean_confidence_correct += count_flag
                sum_sum_confidence_correct += count_flag

                max_reward_path_corrct += count_flag
                max_confidence_path_corrct += count_flag

                max_terminal_reward_correct += count_flag
                max_terminal_confidence_correct += count_flag
                continue

            # 找到具有最高平均奖励路径的输出
            max_reward_path_output = outputs[
                np.argmax(
                    [
                        np.mean([a.reward for a in paths[j]])
                        for j, out in enumerate(outputs)
                    ]
                )
            ]
            max_confidence_path_output = outputs[
                np.argmax(
                    [
                        np.mean([a.confidence for a in paths[j][-1].state])
                        for j, out in enumerate(outputs)
                    ]
                )
            ]

            # 找到最后一个状态具有最高奖励的输出
            max_terminal_reward_output = outputs[
                np.argmax([paths[j][-1].reward for j in range(len(outputs))])
            ]
            max_terminal_confidence_output = outputs[
                np.argmax(
                    [paths[j][-1].state[-1].confidence for j in range(len(outputs))]
                )
            ]

            # 更新最后一个状态最高奖励策略的正确率
            max_terminal_reward_correct += eval_acc(max_terminal_reward_output, answer)
            max_terminal_confidence_correct += eval_acc(
                max_terminal_confidence_output, answer
            )

            # 更新最高平均奖励路径策略的正确率
            max_reward_path_corrct += eval_acc(max_reward_path_output, answer)
            max_confidence_path_corrct += eval_acc(max_confidence_path_output, answer)

            # 使用不同的评分函数更新平均分数和总分数的正确率
            [mean_mean_reward_correct, mean_sum_reward_correct], mean_reward_flag = (
                scores(
                    outputs_set,
                    outputs,
                    paths,
                    answer,
                    mean_mean_reward_correct,
                    mean_sum_reward_correct,
                )
            )
            [sum_mean_reward_correct, sum_sum_reward_correct], sum_reward_flag = scores(
                outputs_set,
                outputs,
                paths,
                answer,
                sum_mean_reward_correct,
                sum_sum_reward_correct,
                scores_function=np.sum,
            )

            (
                [mean_mean_confidence_correct, mean_sum_confidence_correct],
                mean_confidence_flag,
            ) = scores(
                outputs_set,
                outputs,
                paths,
                answer,
                mean_mean_confidence_correct,
                mean_sum_confidence_correct,
            )
            (
                [sum_mean_confidence_correct, sum_sum_confidence_correct],
                sum_confidence_flag,
            ) = scores(
                outputs_set,
                outputs,
                paths,
                answer,
                sum_mean_confidence_correct,
                sum_sum_confidence_correct,
                scores_function=np.sum,
            )

        except Exception as e:
            print(f"{i} not found answer: {e}")  # 捕获异常并打印错误信息
    print(
        "mean_mean_confidence_acc:%f,mean_sum_confidence_acc:%f"
        % (
            mean_mean_confidence_correct / len(mcts_list),
            mean_sum_confidence_correct / len(mcts_list),
        )
    )
    print(
        "sum_mean_confidence_acc:%f,sum_sum_confidence_acc:%f"
        % (
            sum_mean_confidence_correct / len(mcts_list),
            sum_sum_confidence_correct / len(mcts_list),
        )
    )
    print(
        "mean_mean_reward_acc:%f,mean_sum_reward_acc:%f"
        % (
            mean_mean_reward_correct / len(mcts_list),
            mean_sum_reward_correct / len(mcts_list),
        )
    )
    print(
        "sum_mean_reward_acc:%f,sum_sum_reward_acc:%f"
        % (
            sum_mean_reward_correct / len(mcts_list),
            sum_sum_reward_correct / len(mcts_list),
        )
    )
    print(
        "max_terminal_confidence_acc:%f,max_terminal_reward_acc:%f"
        % (
            max_terminal_confidence_correct / len(mcts_list),
            max_terminal_reward_correct / len(mcts_list),
        )
    )
    print(
        "max_reward_path_acc:%f,max_reward_confidence_acc:%f"
        % (
            max_reward_path_corrct / len(mcts_list),
            max_confidence_path_corrct / len(mcts_list),
        )
    )
    print("count_acc:%f,iter:%d" % (count_correct / len(mcts_list), iter))
    print("理论准确率上限:%f" % (1 - len(no_answer_indexs) / len(mcts_list)))
    print("花费时间为 " + str(time.time() - start))
print(model_name)
print(model.__dict__)
print(f"n_action: {n_action}")
print(f"n_confidence: {n_confidence}")
print(f"depth_limit: {depth_limit}")
print(f"cum_reward: {cum_reward.__name__}")  # 打印cum_reward函数的名称
print(f"n_iters: {n_iters}")
