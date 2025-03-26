import jsonlines

data = []  # 使用jsonlines库打开并读取文件
filename = "outputs/Qwen/Qwen2.5-Math-72B-Instruct/math_eval/gaokao_math_qa/test_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl"

import Levenshtein
import numpy as np

options_candidate = ["A", "B", "C", "D"]


def find_box(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def retrieve_chat_aqua_answer(result, options):
    pred = result
    if pred not in options_candidate:
        distances = [Levenshtein.distance(pred, option) for option in options]
        pred = options_candidate[np.argmin(distances)]
    return pred


def eval_acc(output, answer):
    return output == answer


def eval_answer(obj, answer_output, base):
    out = eval(str(answer_output)) / base
    return eval_acc(str(out), str(obj["gt"]))


examples = []
m = 0
n = 0
with jsonlines.open(filename, mode="r") as reader:
    # 遍历文件中的每一行，将其作为字典对象读取
    for obj in reader:
        m += 1
        if obj["score"][0]:
            n += 1
            continue
        if obj["pred"][0] in options_candidate:
            continue
        chocie_text = obj["question"].split("选项:")[-1]
        chocies = []
        for i in range(3):
            word = "(%s)" % options_candidate[i]
            chocies.append(
                word
                + chocie_text.split(word)[-1].split("(%s)" % options_candidate[i + 1])[
                    0
                ]
            )
        word = "(D)"
        chocies.append(word + chocie_text.split(word)[-1])

        pred = retrieve_chat_aqua_answer(find_box(obj["pred"][0]), chocies)
        n += pred == obj["gt"]


print(n / m, n, m)
