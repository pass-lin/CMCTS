# 运行采用qwen-math官方仓库，这里只用eval
# math系列的就直接用他们文章里报道的结果好了
filename = "outputs/Qwen/Qwen2.5-Math-7B-Instruct/math_eval/alg514/test_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl"
import jsonlines
from utils import gsm8k_utils
from sympy import simplify, N
from math import isclose
from utils.gsm8k_utils import math_answer_clean
from reasoners.benchmark import MathEvaluator

evaluator = MathEvaluator(
    output_extractor=gsm8k_utils.retrieve_chat_math_answer,
    filename="data/math_500/test.jsonl",
    sample_prompt_type="rap",
)


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
            answer = math_answer_clean(examples[i]["answer"])
            if "\\text" in answer and answer[:5] != "\\text":
                answer = answer.split("\\text")[0]
            elif "\\text" in answer:
                answer = (
                    answer.split("\\text{")[1][:-1].replace(")", "").replace("(", "")
                )
            answer = evaluator.output_extractor(answer, False)

            pred = evaluator.output_extractor(obj["code"][0])

            right += eval_acc(pred, answer)

        except:
            pass
print(right / total)
