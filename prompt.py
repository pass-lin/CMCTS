weak12k_prompt = {
    "instruction": {
        "role": "system",
        "content": """
下面会给你一个数学数学问题，请一步步思考并解答。每次思考的过程用<think> 和</think>符号圈起来。思考的过程要尽可能的丰富和详细，对每一个内容和细节进行深入地思考，而不是简略的一笔带过，
在你觉得问题思考过程已经能够解决问题后，整理你的思考过程成一分完整答案，并且最终答案写在\\boxed{{}}中。
                  """,
    },
    "useful_examples_prefix": "\n给出一个问题及其对应的子问题和答案。判断这个过程是否有助于回答问题。输出'是'或'否'，并给出理由。\n\n问题 1:%s \n当前的解题过程:\n%s\n",
    "question_prefix": "%s",
    "answer_prefix": "让我们一步步地深入思考和分析这个问题\n",
    "prefix": "\n<think>",
    "stop": "</think>\n",
    "code_prefix": "<think>代码如下所示，我会把最后的计算结果赋值到一个result变量中\n```python",
    "code_stop": "```\n</think>",
    "code_actions": [
        "**我们可以写一段代码来验证我们的想法**",
        "**我们可以写一段代码辅助计算**",
        "**我们可以写一段代码验算我们的计算结果**",
    ],
    "": "",
    "reflect_actions": [
        "**题目应该存在其他的解决方法，我们尝试提出一种不同的解法**",
        "**我们应该一步步检查一下上面的过程是否合理和正确**",
        "**上面的问题可能有错误和不严谨的部分，我们需要一步步对应检查下有没有错误的地方**",
        "**我们需要结合上面的思考过程，是否符合题意**",
        "**题目中有一些细节没有考虑清楚，我们需要检查一下**",
    ],
    "understand_actions": [
        "**我们需要一步步思考来理解题目的含义**",
        "**我们考虑一下题目中是否存在可能的歧义**",
        "**这道题比较难，我们应该先分析下里面提到了什么知识点，利用到了什么数学公式和相关的性质**",
    ],
    "plan_actions": [
        "**需要规划一下我们的解决目标是什么，在解出这个目标前我们可能需要先满足什么条件和对应的子问题**",
        "**可以尝试先给出一个初步的解决方案**",
        "**题目大概分成几个部分，有几个问题需要解决**",
    ],
    "summar_prompt": [
        "**基于上面的思考过程，我们已经解出这道题目。下面我们会整理上面的思考过程并写出最终的解决答案，并且最终答案写在\\boxed{{}}。**",
        "**现在，让我们将我们的思考过程整理成一个完整的答案。并将最终答案写在\\boxed{{}}。**",
    ],
}

game24_prompt = {
    "instruction": {
        "role": "system",
        "content": """
这是一个24点游戏，你需要在下面数字的限定范围内使用加减乘除的方法将其组合成一条24的式子。需要注意的是不能使用没提到的数字，但是可以不使用提到的数字。并且使用的数字不能超过题目里提到的最大次数。
在你觉得问题思考过程已经能够解决问题后，整理你的思考过程成一分完整答案，并且最终的组合式子写在\\boxed{{}}中。
                  """,
    },
    "useful_examples_prefix": "\n这是一个24点游戏，下面会给你一个列表，你使用列表里的数字组合一个式子使得这个式子的答案是24.需要注意的是，数字的使用次数不能大于这个数字在列表里的出现次数，并且不能使用列表中不存在的数字。对于下面的解题过程，如果能帮助解出答案输出'是'，否则输出'否'，又或者他符合24点规则输出'是'，否则输出'否'。\n\n:%s \n当前的解题过程:\n%s\n",
    "question_prefix": "%s",
    "answer_prefix": "让我们一步步地深入思考和分析这个问题\n",
    "prefix": "\n<think>",
    "stop": "</think>\n",
    "code_prefix": "<think>代码如下所示，我会把最后的计算结果赋值到一个result变量中\n```python",
    "code_stop": "```\n</think>",
    "code_actions": [
        "**现在最好的方法是用代码来搜索出来可能的选项**",
        "**可以尝试使用代码去验证这个式子是否符合游戏规则**",
        "**可以试着写一段代码去搜索这个式子**",
    ],
    "": "",
    "reflect_actions": [
        "**我们需要检查一下现在式子有没有出现过没提到的数字或者有数字使用次数超过题目要求**",
        "**我们应该一步步检查一下上面的过程是否是符合游戏规则的**",
        "**我们对着题目一步步检查是否符合要求**",
        "**一步步检查一下我们的答案是不是存在没提到过的数字或者是使用了超出要求的数字**",
        "**上面的问题可能有错误和不严谨的部分，我们需要一步步对应检查下不符合游戏规则的地方**",
        "**总结一下上面的过程，判断一下我们是否有不符合游戏规则的部分并一步步尝试去修正和判断是否符合规则**",
    ],
    "understand_actions": [
        "**我们需要一步步思考来理解规则的含义**",
        "**我们需要一步步梳理一下游戏规则的具体内容**",
        "**我们需要结合游戏规则和题目，思考一下我们目前应该怎么做**"
        "**需要一步步思考一下，目前游戏规则是怎么用的。我们要怎么才能符合规则**"
        
    ],
    "plan_actions": [
        "**需要规划一下我们的解决目标是什么，在解出这个目标前我们可能需要先满足什么条件和对应的子问题**",
        "**可以尝试先给出一个初步的解决方案**",
        "**题目大概分成几个部分，有几个问题需要解决**",
    ],
    "summar_prompt": [
        "**我认为现在的式子已经没问题了，是符合游戏规则的，我们会以latex的格式把这个组合序列写到\\boxed{{}}中。**",
        "**结果详细的思考和检查，我觉得得到的式子已经没有问题了，我们将所获得的组合序列写在\\boxed{{}}作为最终答案。**",
    ],
   "user_prefix":"根据下面的要求使用加减乘除组合将下面的数字成一个计算结果为24的式子，最后把式子写入\\boxed{{}}中,不允许使用下面没提到的数字。\n具体描述：",
}
aqua_prompt = {
    "instruction": {
        "role": "system",
        "content": """
You will be given a mathematics problem. Please think through and solve it step by step. Enclose each thought process with <think> and </think> tags. Make your thought process as rich and detailed as possible, deeply considering every content and detail, rather than briefly skimming over them.
Once you believe the thought process is sufficient to solve the problem, organize your thoughts into a complete answer, and choose the option from “A”, “B”, “C”, “D”, “E” that is closest to your answer. Write the final answer in \\boxed{{}}.
                  """,
    },
    "question_prefix": "%s",
    "answer_prefix": "Let's delve into the problem step by step and analyze it\n",
    "prefix": "\n<think>",
    "stop": "</think>\n",
    "useful_examples_prefix": "\nProvide a question and its corresponding sub-questions and answers. Determine whether this process helps to answer the question. Output 'Yes' or 'No', and give a reason.\n\nQuestion 1:%s \nCurrent problem-solving process:\n%s\n",
    "code_prefix": "<think>Here is the code to verify our idea. I will assign the final calculation result to a variable called result\n```python",
    "code_stop": "```\n</think>",
    "code_actions": [
        "**We can write a piece of code to validate our idea**",
        "**We can write a piece of code to assist with the calculation**",
        "**We can write a piece of code to check our calculation results**",
    ],
    "": "",
    "reflect_actions": [
        "**There should be other methods to solve this problem; let's try to propose a different solution**",
        "**We should step by step check if the above process is reasonable and correct**",
        "**There may be errors and inaccuracies in the above questions; we need to step by step check for any mistakes**",
        "**We need to combine the above thought process to see if it aligns with the problem's intention**",
        "**There are some details in the problem that were not considered clearly; we need to check them**",
    ],
    "understand_actions": [
        "**We need to think step by step to understand the meaning of the problem**",
        "**Let's consider if there are any ambiguities in the problem statement**",
        "**This problem is quite difficult; we should first analyze what knowledge points it involves, what mathematical formulas and related properties it utilizes**",
    ],
    "plan_actions": [
        "**We need to plan what our goal is in solving this, and what conditions and corresponding sub-problems we may need to satisfy first**",
        "**We can try to give a preliminary solution first**",
        "**The problem can be divided into several parts, with several questions that need to be addressed**",
    ],
    "summar_prompt": [
        "**Based on the above thought process, we have solved this problem. First, we will recall the problem, then organize our thought process and write down the final solution, and finally choose the closest answer from “A”, “B”, “C”, “D”, “E”. The final answer will be written in \\boxed{{}}.**",
        "**Now, let's first recall the problem. Then, we will organize our thought process into a complete answer, and finally choose the closest answer from “A”, “B”, “C”, “D”, “E”. The final answer will be written in \\boxed{{}}.**",
    ],
}
school_prompt = {
    "instruction": {
        "role": "system",
        "content": """
下面会给你一个数学数学问题，请一步步思考并解答。每次思考的过程用<think> 和</think>符号圈起来。思考的过程要尽可能的丰富和详细，对每一个内容和细节进行深入地思考，而不是简略的一笔带过，
在你觉得问题思考过程已经能够解决问题后，整理你的思考过程成一分完整答案，如果题目中存在多个问题，每个问题都要依次回答，并且用","隔开。并且最终答案写在\\boxed{{}}中
                  """,
    },
    "question_prefix": "%s",
    "useful_examples_prefix": "\n给出一个问题及其对应的子问题和答案。判断这个过程是否有助于回答问题。输出'是'或'否'，并给出理由。\n\n问题 1:%s \n当前的解题过程:\n%s\n",
    "answer_prefix": "让我们一步步地深入思考和分析这个问题\n",
    "prefix": "\n<think>",
    "stop": "</think>\n",
    "code_prefix": "<think>代码如下所示，我会把最后的计算结果赋值到一个result变量中\n```python",
    "code_stop": "```\n</think>",
    "code_actions": [
        "**我们可以写一段代码来验证我们的想法**",
        "**我们可以写一段代码辅助计算**",
        "**我们可以写一段代码验算我们的计算结果**",
    ],
    "": "",
    "reflect_actions": [
        "**题目应该存在其他的解决方法，我们尝试提出一种不同的解法**",
        "**我们应该一步步检查一下上面的过程是否合理和正确**",
        "**上面的问题可能有错误和不严谨的部分，我们需要一步步对应检查下有没有错误的地方**",
        "**我们需要结合上面的思考过程，是否符合题意**",
        "**题目中有一些细节没有考虑清楚，我们需要检查一下**",
    ],
    "understand_actions": [
        "**我们需要一步步思考来理解题目的含义**",
        "**我们考虑一下题目中是否存在可能的歧义**",
        "**这道题比较难，我们应该先分析下里面提到了什么知识点，利用到了什么数学公式和相关的性质**",
    ],
    "plan_actions": [
        "**需要规划一下我们的解决目标是什么，在解出这个目标前我们可能需要先满足什么条件和对应的子问题**",
        "**针对这道题目，我们大概规划一下有几种解题方案可以尝试一下**",
        "**我们尝试将题目分成多个子问题，然后针对每个子问题给出对应的解答**",
    ],
    "summar_prompt": [
        "**基于上面的思考过程，我们已经解出这道题目。下面我们会整理上面的思考过程并写出最终的解决答案，并且最终答案写在\\boxed{{}}。如果题目中存在多个问题，我会依次回答每个问题，并且用‘,‘隔开。**",
        "**现在，让我们将我们的思考过程整理成一个完整的答案。并将最终答案写在\\boxed{{}}。如果题目中存在多个问题，我会依次回答每个问题，并且用‘,‘隔开。**",
    ],
}
gaokao_prompt = {
    "instruction": {
        "role": "system",
        "content": """
下面会给你一个数学数学问题，请一步步思考并解答。每次思考的过程用<think> 和</think>符号圈起来。思考的过程要尽可能的丰富和详细，对每一个内容和细节进行深入地思考，而不是简略的一笔带过，
在你觉得问题思考过程已经能够解决问题后，首先复述一下题目，然后整理你的思考过程成一份完整答案，在“A”, “B”, “C”, “D”四个选项中选择出正确的答案。。如果题目中存在多个问题，每个问题都要依次回答，并且用","隔开。并且最终答案写在\\boxed{{}}中。
                  """,
    },
    "question_prefix": "%s",
    "answer_prefix": "让我们一步步地深入思考和分析这个问题\n",
    "prefix": "\n<think>",
    "stop": "</think>\n",
    "code_prefix": "<think>代码如下所示，我会把最后的计算结果赋值到一个result变量中\n```python",
    "code_stop": "```\n</think>",
    "useful_examples_prefix": "\n给出一个问题及其对应的子问题和答案。判断这个过程是否有助于回答问题。输出'是'或'否'，并给出理由。\n\n问题 1:%s \n当前的解题过程:\n%s\n",
    "code_actions": [
        "**我们可以写一段代码来验证我们的想法**",
        "**我们可以写一段代码辅助计算**",
        "**我们可以写一段代码验算我们的计算结果**",
    ],
    "": "",
    "reflect_actions": [
        "**题目应该存在其他的解决方法，我们尝试提出一种不同的解法**",
        "**我们应该一步步检查一下上面的过程是否合理和正确**",
        "**上面的问题可能有错误和不严谨的部分，我们需要一步步对应检查下有没有错误的地方**",
        "**我们需要结合上面的思考过程，是否符合题意**",
        "**题目中有一些细节没有考虑清楚，我们需要检查一下**",
    ],
    "understand_actions": [
        "**我们需要一步步思考来理解题目的含义**",
        "**我们考虑一下题目中是否存在可能的歧义**",
        "**这道题比较难，我们应该先分析下里面提到了什么知识点，利用到了什么数学公式和相关的性质**",
    ],
    "plan_actions": [
        "**需要规划一下我们的解决目标是什么，在解出这个目标前我们可能需要先满足什么条件和对应的子问题**",
        "**可以尝试先给出一个初步的解决方案**",
        "**题目大概分成几个部分，有几个问题需要解决**",
    ],
    "summar_prompt": [
        "**基于上面的思考过程，我们已经解出这道题目。下面我们先复述一下题目，然后整理上面的思考过程并写出最终的解决答案，在“A”, “B”, “C”, “D”四个选项中选择出正确的答案。并且最终答案写在\\boxed{{}}。如果题目中存在多个问题，我会依次回答每个问题，并且用‘,‘隔开。**",
        "**现在，让我们先复述一下题目，然后将我们的思考过程整理成一个完整的答案，在“A”, “B”, “C”, “D”四个选项中选择出正确的答案。。并将最终答案写在\\boxed{{}}。如果题目中存在多个问题，我会依次回答每个问题，并且用‘,‘隔开。**",
    ],
}
gaokaoen_prompt = {
    "instruction": {
        "role": "system",
        "content": """
Below is a math problem for you to solve step by step. Enclose your thought process with the <think> and </think> tags. Your thought process should be as comprehensive and detailed as possible, delving deeply into each content and detail rather than just skimming over it. 
Once you believe your thought process is sufficient to solve the problem, organize your thoughts into a complete answer. If there are multiple questions in the problem, answer each one in turn, separated by commas, and write the final answer in \boxed{ }. 
                  """,
    },
    "question_prefix": "%s",
    "useful_examples_prefix": "\nProvide a question and its corresponding sub-questions and answers. Determine whether this process helps to answer the question. Output 'Yes' or 'No', and give a reason.\n\nQuestion 1:%s \nCurrent problem-solving process:\n%s\n",
    "answer_prefix": "Let's delve into this problem step by step\n",
    "prefix": "\n<think>",
    "stop": "</think>\n",
    "code_prefix": "<think>Here is the code to validate our thoughts. I will assign the final calculation result to a variable called result\n```python",
    "code_stop": "```\n</think>",
    "code_actions": [
        "**We can write a piece of code to verify our idea**",
        "**We can write a piece of code to assist with the calculations**",
        "**We can write a piece of code to check our calculation results**",
    ],
    "": "",
    "reflect_actions": [
        "**There should be other methods to solve this problem; let's try to propose a different solution**",
        "**We should step by step check if the above process is reasonable and correct**",
        "**There may be errors and inaccuracies in the above questions; we need to step by step check for any mistakes**",
        "**We need to combine the above thought process to see if it aligns with the problem's intention**",
        "**There are some details in the problem that were not considered clearly; we need to check them**",
    ],
    "understand_actions": [
        "**We need to think step by step to understand the meaning of the problem**",
        "**Let's consider if there are any ambiguities in the problem statement**",
        "**This problem is quite difficult; we should first analyze what knowledge points it involves, what mathematical formulas and related properties it utilizes**",
    ],
    "plan_actions": [
        "**We need to plan what our goal is in solving this, and what conditions and corresponding sub-problems we may need to satisfy first**",
        "**We can try to give a preliminary solution first**",
        "**The problem can be divided into several parts, with several questions that need to be addressed**",
    ],
    "summar_prompt": [
        "**Based on the above thought process, we have solved this problem. Now, let's organize our thoughts into a complete answer and write the final answer in \\boxed{}. If there are multiple questions, I will answer each one in turn, separated by commas.**",
        "**Now, let's compile our thought process into a complete answer and place the final answer in \\boxed{}. If there are multiple questions, I will answer each one in turn, separated by commas.**",
    ],
}
math_prompt = {
    "instruction": {
        "role": "system",
        "content": """
Below is a math problem for you to solve step by step. Enclose your thought process with the <think> and </think> tags. Your thought process should be as comprehensive and detailed as possible, delving deeply into each content and detail rather than just skimming over it. 
Once you believe your thought process is sufficient to solve the problem, organize your thoughts into a complete answer. If there are multiple questions in the problem, answer each one in turn, separated by commas, and write the final answer in \boxed{ }. 
                  """,
    },
    "question_prefix": "%s",
    "useful_examples_prefix": "\nProvide a question and its corresponding sub-questions and answers. Determine whether this process helps to answer the question. Output 'Yes' or 'No', and give a reason.\n\nQuestion 1:%s \nCurrent problem-solving process:\n%s\n",
    "answer_prefix": "Let's delve into this problem step by step\n",
    "prefix": "\n<think>",
    "stop": "</think>\n",
    "code_prefix": "<think>Here is the code to validate our thoughts. I will assign the final calculation result to a variable called result\n```python",
    "code_stop": "```\n</think>",
    "code_actions": [
        "**We can write a piece of code to verify our idea**",
        "**We can write a piece of code to assist with the calculations**",
        "**We can write a piece of code to check our calculation results**",
    ],
    "": "",
    "reflect_actions": [
        "**There should be other ways to solve the problem; we can try to propose a different solution.**",
        "**We need to check the process above step by step to see if it is reasonable and correct.**",
        "**There are errors in the problem above; we need to check each step carefully to avoid mistakes.**",
        "**The solution process above might be wrong; we can propose a new solution to verify it.**",
        "**There seem to be contradictions in the previous thinking, which need to be reorganized.**",
    ],
    "understand_actions": [
        "**We need to think step by step to understand the meaning of the problem**",
        "**Let's consider if there are any ambiguities in the problem statement**",
        "**This problem is quite difficult; we should first analyze what knowledge points it involves, what mathematical formulas and related properties it utilizes**",
    ],
    "plan_actions": [
        "**We need to plan what our solution goal is. Before achieving this goal, we might need to meet certain conditions and address corresponding sub-problems.**",
        "**For this problem, let's roughly plan how many solution approaches we can try.**",
        "**We can try to break down the problem into several sub-problems and then provide solutions for each sub-problem.**",
    ],
    "summar_prompt": [
        "**Based on the above thought process, we have solved this problem. Now, let's organize our thoughts into a complete answer and write the final answer in \\boxed{}. If there are multiple questions, I will answer each one in turn, separated by commas.**",
        "**Now, let's compile our thought process into a complete answer and place the final answer in \\boxed{}. If there are multiple questions, I will answer each one in turn, separated by commas.**",
    ],
}
svamp_prompt = {
    "instruction": {
        "role": "system",
        "content": """
Below is a mathematical problem. Please think step by step and solve it. Enclose each thought process with the <think> and </think> symbols. The thought process should be as rich and detailed as possible, delving into every content and detail deeply, rather than just skimming over,
After you feel that the thought process is sufficient to solve the problem, organize your thought process into a complete answer, and write the final answer in \\boxed{{}}.
                  """,
    },
    "question_prefix": "%s",
    "answer_prefix": "Let's delve into the problem step by step and analyze it\n",
    "prefix": "\n<think>",
    "useful_examples_prefix": "\nProvide a question and its corresponding sub-questions and answers. Determine whether this process helps to answer the question. Output 'Yes' or 'No', and give a reason.\n\nQuestion 1:%s \nCurrent problem-solving process:\n%s\n",
    "stop": "</think>\n",
    "code_prefix": "<think>Here is the code to verify our idea. I will assign the final calculation result to a variable called result\n```python",
    "code_stop": "```\n</think>",
    "code_actions": [
        "**We can write a piece of code to validate our idea**",
        "**We can write a piece of code to assist with the calculation**",
        "**We can write a piece of code to check our calculation results**",
    ],
    "": "",
    "reflect_actions": [
        "**There should be other methods to solve this problem; let's try to propose a different solution**",
        "**We should step by step check if the above process is reasonable and correct**",
        "**There may be errors and inaccuracies in the above questions; we need to step by step check for any mistakes**",
        "**We need to combine the above thought process to see if it aligns with the problem's intention**",
        "**There are some details in the problem that were not considered clearly; we need to check them**",
    ],
    "understand_actions": [
        "**We need to think step by step to understand the meaning of the problem**",
        "**Let's consider if there are any ambiguities in the problem statement**",
        "**This problem is quite difficult; we should first analyze what knowledge points it involves, what mathematical formulas and related properties it utilizes**",
    ],
    "plan_actions": [
        "**We need to plan what our goal is in solving this, and what conditions and corresponding sub-problems we may need to satisfy first**",
        "**We can try to give a preliminary solution first**",
        "**The problem can be divided into several parts, with several questions that need to be addressed**",
    ],
    "summar_prompt": [
        "**Based on the above thought process, we have solved this problem. Below we will organize the above thought process and write the final solution, and write the final answer in \\boxed{{}}.**",
        "**Now, let's organize our thought process into a complete answer. And write the final answer in \\boxed{{}}.**",
    ],
}
