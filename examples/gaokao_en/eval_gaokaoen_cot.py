# 运行采用qwen-math官方仓库，这里只用eval
# math系列的就直接用他们文章里报道的结果好了
filename = "outputs/Qwen/Qwen2.5-7B-Instruct/math_eval/gaokao2023en/test_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl"
import jsonlines
from utils import gsm8k_utils
from sympy import simplify, N
from math import isclose
import string
from reasoners.benchmark import MathEvaluator

evaluator = MathEvaluator(
    output_extractor=gsm8k_utils.retrieve_chat_gaokaoen_answer,
    filename="data/gaokao2023en/test.jsonl",
    sample_prompt_type="rap",
)
dataset = list(evaluator.full_dataset)[:]


def eval_acc(a, b):
    if type(a) == list or type(b) == list:
        a = list(a)
        b = list(b)
    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:

        def caculate(a, b):
            if len(a) != len(b):
                return False
            for i in range(len(a)):
                if not isclose(float(N(a[i])), float(N(b[i])), rel_tol=1e-4):
                    return False
            return True

        if type(a) == list:
            if any([caculate(a, b), caculate(a, b[::-1])]):
                return True
        elif isclose(float(N(a)), float(N(b)), rel_tol=1e-4):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


examples = list(evaluator.full_dataset)
total = 0
right = 0
with jsonlines.open(filename, mode="r") as reader:
    # 遍历文件中的每一行，将其作为字典对象读取
    for i, obj in enumerate(reader):
        total += 1
        try:
            options = None
            if examples[i]["answer"][:3] == "$x=":
                examples[i]["answer"] = examples[i]["answer"][3:]
            if examples[i]["answer"] in string.ascii_uppercase:
                option_label = ""
                choice_string = ""
                options = []
                for temp_str in examples[i]["question"].split(":")[-1].split("("):
                    if len(temp_str) < 2:
                        continue
                    elif temp_str[1] == ")":
                        if option_label != "":
                            options.append(
                                [
                                    option_label,
                                    choice_string.replace("$", "").replace(" ", ""),
                                ]
                            )
                        option_label = temp_str[0]
                        choice_string = temp_str[2:]
                    else:
                        choice_string += temp_str
                options.append(
                    [option_label, choice_string.replace("$", "").replace(" ", "")]
                )

            pred = evaluator.output_extractor(obj["code"][0], options=options)
            answer = evaluator.output_extractor(examples[i]["answer"], False, options)
            right += eval_acc(pred, answer)

        except:
            pass
print(right / total)
