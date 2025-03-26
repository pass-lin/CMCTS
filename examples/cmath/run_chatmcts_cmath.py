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

from LLM import ChatCOTModel
from rap_prompt import weak12kprompt as prompt
from rap_prompt import weak12k_usefulprompt as useful_prompt

model = ChatCOTModel(
    model_name,
    prompt=prompt,
    stop=prompt["stop"],
    select_tokens=[" 是", " 否"],
    top_p=0.7,
    top_k=50,
    max_tokens=512,
    max_model_len=4096,
    temperature=1.05,
)

n_action: int = 8
n_confidence: int = 3
depth_limit: int = 8
cum_reward = np.sum
n_iters: int = 9


from LLM import *
from reasoners.benchmark import CMathEvaluator
from reasoners import Reasoner
from reasoners.algorithm import MCTS, MiddleResult
from world_model import Weak12KWorldModel
from search_config import Weak12KConfig
from utils import gsm8k_utils
from copy import deepcopy
from tqdm import tqdm


search_algo_params = {}
evaluator = CMathEvaluator(
    output_extractor=gsm8k_utils.retrieve_chat_weak12k_answer,
    answer_extractor=lambda x: x["answer"],
    filename="data/cmath/test.jsonl",
    init_prompt=prompt,
    sample_prompt_type="rap",
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
    )
):
    try:
        world_model = Weak12KWorldModel(n_confidence=n_confidence)
        config = Weak12KConfig(
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

        for i, state in enumerate(states):
            if state == "get_action":
                if len(middle_results[i].action_outputs) == 0:
                    middle_results[i].action_outputs = [
                        prompt["overall_question_prefix"] + dataset[i]["question"]
                    ]
                for j in range(len(middle_results[i].action_outputs)):
                    if "现在我们" in middle_results[i].action_outputs[j]:
                        middle_results[i].action_outputs[j] = (
                            prompt["overall_question_prefix"]
                            + dataset[i]["question"]
                            + prompt["question_postfix"]
                            + "\n 我们需要将最终答案写在\\boxed{{}}。"
                        )
        steps += 1

    def eval_acc(output, answer):
        try:
            answer = eval(answer)
            output = abs(
                eval(output)
            )  # 这个数据集没有负数的答案，但是题目可能让题目回答负数，比如水底下50米表达是-50，但答案是50
            flags = []
            for t in [
                output / 100,
                output,
                output * 100,
            ]:  # 考虑百分数的不同回答情况，当然，baseline也是有做出相应修正的
                # 这里的精度取1e-2，因为大模型经常会四舍五入到小数点三位，baseline也会做出对应的修正
                flags.append(abs(t - answer) <= 1e-2)
            return any(flags)
        except:
            pass
        return False

    def scores(
        outputs_set,
        outputs,
        trace_in_each_iter,
        answer,
        mean_mean_reward_correct,
        mean_sum_reward_correct,
        scores_function=np.mean,
        reward2score=True,
        return_flag=False,
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
        # 初始化分数列表，对个可能的输出
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
        acc_flag_1 = eval_acc(mean_reward_output, answer)
        mean_mean_reward_correct += acc_flag_1
        # 更新总分数方法的累计正确率
        acc_flag_2 = eval_acc(sum_reward_output, answer)
        mean_sum_reward_correct += acc_flag_2
        return [mean_mean_reward_correct, mean_sum_reward_correct], [
            acc_flag_1,
            acc_flag_2,
        ]

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
    total = 0
    for i in range(len(mcts_list)):
        flag = True
        try:
            if (
                examples[i]["question"].count("?") > 2
                or "提出" in examples[i]["question"]
            ):
                # 会存在一些奇葩题目在这个数据集里，要做修正
                # 还有比如”你再提出一个问题巴拉巴拉“这种奇葩问题用规则过滤一下
                # 之所以在这里过滤而不在选择数据那里过滤，是因为在那里过滤性能会莫名其妙下降，我也不知为什么
                continue
            total += 1
            # 提取当前示例的答案
            answer = examples[i]["answer"]
            outputs = []
            if int(answer) == float(answer):
                answer = str(int(answer))  # 如果答案是整数，转换为整数字符串
            else:
                answer = str(answer)  # 否则保持浮点数字符串
            # 提取每次迭代中的输出
            paths = []
            for t in mcts_list[i].trace_in_each_iter:
                try:
                    if "\\boxed" in t[-1].state[-1].sub_answer.replace("\\boxed{}", ""):
                        output = evaluator.output_extractor(t[-1].state)
                    else:
                        output = evaluator.output_extractor(
                            "".join(t.sub_answer for t in t[-1].state)
                        )
                    output = str(output)
                    assert eval(output) != None  # 尝试评估输出是否为有效表达式
                except:
                    try:
                        # 如果无效，尝试从子问题中提取输出并评估
                        output = str(
                            evaluator.output_extractor(
                                "\n\n".join(
                                    [
                                        a.sub_question + "\n" + a.sub_answer
                                        for a in t[-1].state[:-1]
                                    ]
                                )
                                + t[-1].state[-1].sub_question
                            )
                        )
                        eval(output)
                    except:
                        continue  # 如果仍然无效，跳过该路径
                if output == "None" or output == None:
                    continue
                outputs.append(str(output))
                paths.append(t)

            # 如果没有输出且搜索空间为空，则重新初始化MCTS实例
            # if len(outputs) == 0 and mcts_list[i].is_no_search_space():
            #    mcts_list[i].initial(mcts_list[i].world_model, mcts_list[i].search_config)
            if len(outputs) == 0:
                print(f"{i} not found answer")
                continue
            # 获取唯一的输出集合
            outputs_set = list(set(outputs))

            # 找到出现次数最多的输出
            counts = [outputs.count(a) for a in outputs_set]
            count_output = outputs_set[np.argmax(counts)]

            # 更新计数策略的正确率
            count_flag = eval_acc(count_output, answer)
            count_correct += count_flag
            if not count_flag:
                print(i, answer, outputs)
                print(examples[i]["question"])
                print(paths[outputs.index(count_output)][-1].state[-1].sub_answer)
                print("*" * 20)
            # 如果答案不在输出集中，记录索引
            if answer not in outputs_set:
                no_answer_indexs.append(i)

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
        % (mean_mean_confidence_correct / total, mean_sum_confidence_correct / total)
    )
    print(
        "sum_mean_confidence_acc:%f,sum_sum_confidence_acc:%f"
        % (sum_mean_confidence_correct / total, sum_sum_confidence_correct / total)
    )
    print(
        "mean_mean_reward_acc:%f,mean_sum_reward_acc:%f"
        % (mean_mean_reward_correct / total, mean_sum_reward_correct / total)
    )
    print(
        "sum_mean_reward_acc:%f,sum_sum_reward_acc:%f"
        % (sum_mean_reward_correct / total, sum_sum_reward_correct / total)
    )
    print(
        "max_terminal_confidence_acc:%f,max_terminal_reward_acc:%f"
        % (max_terminal_confidence_correct / total, max_terminal_reward_correct / total)
    )
    print(
        "max_reward_path_acc:%f,max_reward_confidence_acc:%f"
        % (max_reward_path_corrct / total, max_confidence_path_corrct / total)
    )
    print("count_acc:%f,iter:%d" % (count_correct / total, iter))
    print("花费时间为 " + str(time.time() - start))
print(model_name)
print(model.__dict__)
print(f"n_action: {n_action}")
print(f"n_confidence: {n_confidence}")
print(f"depth_limit: {depth_limit}")
print(f"cum_reward: {cum_reward.__name__}")  # 打印cum_reward函数的名称
print(f"n_iters: {n_iters}")
