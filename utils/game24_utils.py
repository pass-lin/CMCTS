
from utils.gsm8k_utils import *
def retrieve_chat_game24_answer(result):
    if isinstance(result, list):
        result = result[-1].sub_answer
    out = find_box(
        "boxed"
        + result.replace("\x08", "b")
        .replace("{{}}", "{}")
        .replace("boxed{}", "")
        .split("boxed")[-1]
    )
    return latex2sympy(out)

def eval_acc(count_output, candidate):
    import re
    try:
        
        path = str(count_output)
        answer = count_output.evalf()
        if float(answer) != 24:
            return False
        candidate = sorted([int(t) for t in candidate])[::-1]
        path = path.replace("-1*", "")  
        for num in candidate:
            path = path.replace(str(num), "",1)
        
        return not bool(re.search(r'\d', path))
        
    except:
        pass
    return False