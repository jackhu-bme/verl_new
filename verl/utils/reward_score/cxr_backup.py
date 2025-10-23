# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re

# from mathruler.grader import extract_boxed_content #, grade_answer

def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"

def is_consistent(predict_str: str, ground_truth: str) -> bool:
    """Check if the predicted string is consistent with the ground truth."""
    # This function can be customized based on how you want to check consistency.
    # For example, you might want to check if the predict_str contains the ground_truth.
    
    # capital to lower
    predict_str = predict_str.lower()
    ground_truth = ground_truth.lower()
    res = predict_str.startswith(ground_truth)
    return res



def format_reward(predict_str: str) -> float:
    # pattern_one = re.compile(r"<think>.*</think>", re.DOTALL)
    # match_one = re.fullmatch(pattern_one, predict_str)
    match_two = re.search(r"\\boxed\{(.*?)\}", predict_str, re.DOTALL)
    # match_result = (match_one is not None) and (match_two is not None)
    match_result = (match_two is not None)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_content(predict_str)
        # print(f"-" * 20)
        # print(f"Answer after extract: {answer}")
        # print(f"Ground Truth: {ground_truth}")
        # print(f"-" * 20)
    else:
        answer = predict_str
    return 1.0 if is_consistent(answer, ground_truth) else 0.0

# import re

def tool_reward(predict_str: str) -> float:
    tool_success_tag = "You successfully used the crop tool!"
    matches = re.findall(re.escape(tool_success_tag), predict_str)

    if len(matches) == 1:
        return 1.0
    elif len(matches) > 1:
        return 0.5
    else:
        return 0.0


def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1, tool_score: float = 0.1) -> float:
    breakpoint()
    tool_r = tool_reward(predict_str)
    acc_r = acc_reward(predict_str, ground_truth, use_boxed)
    format_r = format_reward(predict_str)
    # not sure whether this is needed
    if tool_r < 0.5:
        acc_r = 0 # if no tool use, then no acc reward
        format_r = 0.0
    return (1.0 - format_score - tool_score) * acc_r + format_score * format_r + tool_score * tool_r