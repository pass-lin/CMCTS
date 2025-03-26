import re
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy
import Levenshtein
import string


def math_answer_clean(text):
    text = (
        text.replace("\\end{pmatrix}", "")
        .replace("\\begin{pmatrix}", "")
        .replace("\\pi", "\pi")
        .replace("\\right", "")
        .replace("\left", "")
        .replace("\right", "")
        .replace("\\$", "")
        .replace("^{\circ}", "")
        .replace("^\\circ", "")
        .replace("\mbox{ cm}^2", "")
        .replace("\\\\\\", ",\\")
        .replace("\\\\", ",")
        .replace(" ", "")
    )
    if "\\pm" in text:
        pre, post = text.split("\\pm")
        text = [pre + "+" + post, pre + "-" + post]
    return text


def school_answer_clean(text):
    return (
        text.replace("\\pi", "\pi")
        .replace("\\right", "")
        .replace("\left", "")
        .replace("\right", "")
        .replace("\\$", "")
        .replace("^{\circ}", "")
        .replace("^\\circ", "")
        .replace("\mbox{ cm}^2", "")
        .replace(" ", "")
    )


def _parse(s):
    for f in [latex2sympy, parse_latex, parse_expr]:
        try:
            return f(s.replace("\\\\", "\\"))
        except:
            try:
                return f(s)
            except:
                pass
    return s


from typing import Optional, Union
import numpy as np
from reasoners.base import AlgorithmOutput
from copy import deepcopy
from utils.parser import extract_answer


def retrieve_answer(output: Union[list, str, AlgorithmOutput]) -> Optional[str]:
    """
    output should be a world_model.GSM8kState if being a list
    """
    if isinstance(output, AlgorithmOutput):
        if (result := getattr(output, "aggregated_result", None)) is not None:
            return result
        output = output.terminal_state
    if isinstance(output, list):
        output = output[-1].sub_answer
    match = re.match(r".*The answer is .*?([ $.0-9,\-=]+).*\..*", output)
    if match is None:
        return None
    answer = match[1].replace(",", "").replace("$", "").replace(" ", "")
    if "=" in answer:
        answer = answer[answer.rindex("=") + 1 :]
    return answer


def retrieve_code_answer(output):
    """
    output should be a world_model.GSM8kState if being a list
    """
    if isinstance(output, AlgorithmOutput):
        if (result := getattr(output, "aggregated_result", None)) is not None:
            return result
        output = output.terminal_state[-1].sub_answer
    if isinstance(output, list):
        output = output[-1].sub_answer
    output = output.split("```")[-1].lower().split("the answer is")[-1][:-1]
    try:
        result = eval(output)
        if isinstance(result, tuple):
            return result[-1]
        return result
    except:
        return None


from LLM import execute


def state2code(output, code_prompt):
    code = code_prompt["inital_variable"]
    for i, t in enumerate(output):
        q, a = t[:2]
        code += code_prompt["useful_question_prefix"] % (i + 2) + q + "\n" + a + "\n"
    return code


def retrieve_gsm8k_code_answer(output, code_prompt):
    local_vars = {}
    execute(state2code(output, code_prompt), local_vars)
    return local_vars.get("result")


def retrieve_chat_answer(output):
    if isinstance(output, AlgorithmOutput):
        if (result := getattr(output, "aggregated_result", None)) is not None:
            return result
        output = output.terminal_state[-1].sub_answer
    if isinstance(output, list):
        output = output[-1].sub_answer
    answer_output = extract_answer(output, "gsm8k")
    answer_output = answer_output.split("=")[-1]
    output = ""
    for s in answer_output:
        if s.isdigit() or s == ".":
            output += s
    return output


def retrieve_codeweak12k_answer(result):
    out = result[-1].sub_answer
    out = out[:-1].split("答案是")[-1]
    out = eval(out)
    return out


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


def fullwidth_to_halfwidth(text):
    # 全角字符和半角字符的ASCII码差值
    offset = 0xFEE0
    # 空格的特殊处理
    space_offset = 0x20

    result = []
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            # 转换全角字符到半角字符
            result.append(chr(code - offset))
        elif code == 0x3000:
            # 转换全角空格到半角空格
            result.append(chr(code - space_offset))
        else:
            # 保持其他字符不变
            result.append(char)
    return "".join(result)


def retrieve_chat_middleschool_answer(result):
    if isinstance(result, list):
        result = result[-1].sub_answer
    result = (
        result.replace("，", ",")
        .replace("。", "")
        .replace(" ", "")
        .replace("\\text{且}", ",")
    )
    result = fullwidth_to_halfwidth(result)
    out = find_box(
        "boxed"
        + result.replace("\x08", "b")
        .replace("{{}}", "{}")
        .replace("boxed{}", "")
        .split("boxed")[-1]
    )
    option2 = ["A.", "B.", "C.", "D."]
    if out in ["A", "B", "C", "D"]:
        return out, "option"
    elif any([t in out for t in option2]):
        distances = [Levenshtein.distance(out, option) for option in option2]
        return ["A", "B", "C", "D"][np.argmin(distances)], "option"
    try:
        sympy_expr = latex2sympy(school_answer_clean(out))
        return sympy_expr, "sympy"
    except:
        out = "".join(filter(lambda x: not "\u4e00" <= x <= "\u9fff", out))
        return out, "string"


def gaokaoen_answer_clean(text):
    text = text.split("=")[-1]
    text = (
        text.replace("am", ",")
        .replace("a.m.", ",")
        .replace("\\%", "")
        .replace("%", "")
        .replace("cm^{2}", "")
        .replace("m^{2}", "")
        .replace("solution", "")
        .replace("$", "")
        .replace("and", ",")
        .replace("\\end{pmatrix}", "")
        .replace("\\begin{pmatrix}", "")
        .replace("\\pi", "\pi")
        .replace("\\right", "")
        .replace("\left", "")
        .replace("\right", "")
        .replace("\\$", "")
        .replace("^{\circ}", "")
        .replace("^\\circ", "")
        .replace("\mbox{ cm}^2", "")
        .replace("\\\\\\", ",\\")
        .replace("\\\\", ",")
        .replace(" ", "")
    )
    if "\\pm" in text:
        pre, post = text.split("\\pm")
        text = [pre + "+" + post, pre + "-" + post]
    if "\\frac" in text and "\\frac{" not in text:
        pre, post = text.split("\\frac")
        a, b, post = post[0], post[1], post[2:]
        text = pre + "\\frac{" + a + "}{" + b + "}" + post
    if text[0] == "(" and text[-1] == ")":
        return text[1:-1]
    return text


def retrieve_chat_alg514_answer(result):
    if isinstance(result, list):
        result = result[-1].sub_answer

    split = result.split("\\boxed")
    if len(split) > 2:
        out = []
        for i in range(1, len(split)):
            out.append(retrieve_chat_alg514_answer("\\boxed" + split[i]))
        return out

    def int_answer(answer):
        for j in range(len(answer)):
            if int(answer[j]) == float(answer[j]):
                answer[j] = int(answer[j])
        return answer

    def remove_text_commands(s):
        pattern = r"\\text\{.*?\}"
        return re.sub(pattern, "", s)

    result = find_box(
        "boxed"
        + result.replace("\x08", "b")
        .replace("{{}}", "{}")
        .replace("boxed{}", "")
        .split("boxed")[-1]
    )
    result = (
        result.replace(",00", "00")
        .replace("%", "")
        .replace(" ", "")
        .replace("\\text", ",\\text")
    )
    result = remove_text_commands(result)

    while ",," in result:
        result = result.replace(",,", ",")
    if "\\frac" in result:
        result = str(latex2sympy(result).n())
    else:
        out = parse_expr(result)
    if isinstance(out, list) or isinstance(out, tuple):
        return int_answer(list(out))

    result = eval(result)
    if int(result) == float(result):
        return int(result)
    return float(result)


def retrieve_chat_gaokaoen_answer(result, find_box_flag=True, options=None):
    if find_box_flag:
        if isinstance(result, list):
            result = result[-1].sub_answer
        result = (
            result.replace("，", ",")
            .replace("。", "")
            .replace(" ", "")
            .replace("\\text{且}", ",")
        )
        result = fullwidth_to_halfwidth(result)
        result = find_box(
            "boxed"
            + result.replace("\x08", "b")
            .replace("{{}}", "{}")
            .replace("boxed{}", "")
            .split("boxed")[-1]
        )
        if "\\text" in result and result[:5] == "\\text":
            result = result[6:-1]
        elif "\text" in result and result[:4] == "\text":
            result = result[5:-1]
    if options is not None:
        if result in [t[0] for t in options]:
            return result
        distances = [Levenshtein.distance(result, option[1]) for option in options]
        return options[np.argmin(distances)][0]
    if isinstance(result, list):
        out = [_parse(gaokaoen_answer_clean(t)) for t in result]
    else:
        out = _parse(gaokaoen_answer_clean(result))
    if type(out) == list:
        out = [_parse(t) for t in out]
        for i in range(len(out)):
            try:
                out[i] = out[i].simplify()
            except:
                pass
    return out


def retrieve_chat_math_answer(result, find_box_flag=True):
    if find_box_flag:
        if isinstance(result, list):
            result = result[-1].sub_answer
        result = (
            result.replace("，", ",")
            .replace("。", "")
            .replace(" ", "")
            .replace("\\text{且}", ",")
        )
        result = fullwidth_to_halfwidth(result)
        result = find_box(
            "boxed"
            + result.replace("\x08", "b")
            .replace("{{}}", "{}")
            .replace("boxed{}", "")
            .split("boxed")[-1]
        )
        if "\\text" in result and result[:5] == "\\text":
            result = result[6:-1]
        elif "\text" in result and result[:4] == "\text":
            result = result[5:-1]
    if isinstance(result, list):
        out = [_parse(math_answer_clean(t)) for t in result]
    else:
        out = _parse(math_answer_clean(result))
    if type(out) == list:
        out = [_parse(t) for t in out]
        for i in range(len(out)):
            try:
                out[i] = out[i].simplify()
            except:
                pass
    return out


def retrieve_chat_gaokao_answer(result, options):
    if isinstance(result, list):
        result = result[-1].sub_answer
    options = [value for value in options.values()]
    result = (
        result.replace("，", ",")
        .replace("。", "")
        .replace(" ", "")
        .replace("\\text{且}", ",")
    )
    result = fullwidth_to_halfwidth(result)
    pred = find_box(
        "boxed"
        + result.replace("\x08", "b")
        .replace("{{}}", "{}")
        .replace("boxed{}", "")
        .split("boxed")[-1]
    ).replace(" ", "")
    options_candidate = ["A", "B", "C", "D"] + [
        "AB",
        "ABC",
        "ABCD",
        "AC",
        "AD",
        "ACD",
        "BC",
        "BCD",
        "BD",
        "CD",
    ]
    if pred.replace(",", "") not in options_candidate:
        distances = [Levenshtein.distance(pred, option) for option in options]
        pred = options_candidate[np.argmin(distances)]
    return pred.replace(",", "")


def retrieve_chat_weak12k_answer(result):
    if isinstance(result, list):
        result = result[-1].sub_answer
    out = find_box(
        "boxed"
        + result.replace("\x08", "b")
        .replace("{{}}", "{}")
        .replace("boxed{}", "")
        .split("boxed")[-1]
    )
    answer_output = (
        out.split("\\text{ and }")[-1]
        .split(",")[-1]
        .split("，")[-1]
        .split("或")[-1]
        .replace("\\pi", "*3.14")
        .replace("pi", "3.14")
        .replace(":", "/")
    )
    answer_output = (
        answer_output.replace("frac", "\\frac")
        .replace("%", "/100")
        .replace("pi", "3.14")
    )
    answer_output = answer_output.split("=")[-1]
    answer_output = "".join(
        filter(lambda x: not "\u4e00" <= x <= "\u9fff", answer_output)
    )
    if "frac" in answer_output:
        output = str(simplify(_parse(answer_output.replace("{}", "{1}"))).evalf())
        try:
            eval(output)
        except:
            for word in string.ascii_lowercase:
                output = output.replace(word, "1")
    else:
        output = ""
        for s in answer_output:
            if s.isdigit() or s in [".", "+", "-", "*", "/"]:
                output += s
    try:
        if "text{折}" in result and eval(output) > 1:
            output = str(eval(output) / 10)
        return str(eval(output))
    except:
        return None


def retrieve_answer_from_dataset(answer: Union[str, dict]) -> str:
    if isinstance(answer, dict):
        answer = answer["answer"]
    return re.match(r"[\S\s]*#### (.*)$", answer)[1].replace(",", "").replace(" ", "")


def judge_answer(output: Optional[str], answer: str) -> bool:
    if output is None:
        return False
    try:
        output = int(output)
        answer = int(answer)
        return output == answer
    except ValueError:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output == answer
    except ValueError:
        pass
    return output == answer


def get_tot_answer(result):
    match = re.match(r".*The answer is .*?([ $.0-9,\-=]+).*\..*", result.state)
    if match is None:
        return None
    answer = match[1].replace(",", "").replace("$", "").replace(" ", "")
    if "=" in answer:
        answer = answer[answer.rindex("=") + 1 :]
    return answer


def get_tot_gsm8k_predict_answer(results):
    answers_value = []
    answers_num = []
    for result in results:
        answer = get_tot_answer(result)
        if answer == None:
            continue
        if answer in answers_value:
            answers_num[answers_value.index(answer)] += 1
        else:
            answers_num.append(1)
            answers_value.append(answer)
    if len(answers_value) == []:
        return None
    return answers_value[np.argmax(answers_num)]


def get_cot_gsm8k_predict_answer(result):
    match = re.match(r".*The answer is .*?([ $.0-9,\-=]+).*\..*", result)
    if match is None:
        return None
    answer = match[1].replace(",", "").replace("$", "").replace(" ", "")
    if "=" in answer:
        answer = answer[answer.rindex("=") + 1 :]
    return answer


def get_cot_math23k_predict_answer(result):
    pattern = r"答案是\s*([ $.0-9,\-=]+)"
    match = re.search(pattern, result)
    if match is None:
        return None
    answer = (
        match[1]
        .replace(",", "")
        .replace("$", "")
        .replace(" ", "")
        .replace("答案是", "")
    )
    if "=" in answer:
        answer = answer[answer.rindex("=") + 1 :]
    while answer[-1] == ".":
        answer = answer[:-1]
    return answer


def get_mcts_math23k_predict_answer(result):
    result = deepcopy(result)
    t = result.terminal_state[-1]
    answer = (
        t.sub_answer.split("<|end_of_text|>")[0]
        .split("<|endoftext|>")[0]
        .split("\n")[0]
        + "."
    )
    return get_cot_math23k_predict_answer(answer)


def get_mcts_code_math23k_predict_answer(result):
    result = deepcopy(result)
    t = result.terminal_state[-1]
    answer = t.sub_answer.split("\n")[-1][:-1]
    answer = answer.split("答案是")[-1]
    if "," in answer:
        answer = answer.split(",")[-1]
    return answer


def get_tot_math23k_predict_answer(results, return_answer_list=False):
    answers_value = []
    answers_num = []
    answer_list = []
    for result in results:
        answer = get_cot_math23k_predict_answer(result.state)
        answer_list.append(answer)
        if answer in answers_value:
            answers_num[answers_value.index(answer)] += 1
        elif answer is not None:
            answers_num.append(1)
            answers_value.append(answer)
    if len(answers_value) == []:
        return None
    if return_answer_list:
        return answers_value[np.argmax(answers_num)], answer_list
    return answers_value[np.argmax(answers_num)]
