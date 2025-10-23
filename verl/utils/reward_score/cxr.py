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

TOOL_SUCCESS_TAG = "You successfully used the crop tool!"


def extract_boxed_content(text: str, start_after: int = 0) -> str:
    """
    Extracts answers in \\boxed{} that appear *after* start_after.
    Supports nested braces inside \\boxed{...}.
    """
    # find from the right side
    start_pos = text.rfind(r"\boxed{", start_after)
    if start_pos == -1:
        return "None"

    depth = 0
    content = text[start_pos + len(r"\boxed{") :]
    end_pos = -1
    for i, char in enumerate(content):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth == -1:
            end_pos = i
            break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"


def is_consistent(predict_str: str, ground_truth: str) -> bool:
    """Check if the predicted string is consistent with the ground truth."""
    predict_str = predict_str.lower()
    ground_truth = ground_truth.lower()
    return predict_str.startswith(ground_truth)


def format_reward(predict_str: str) -> float:
    tag_idx = predict_str.rfind(TOOL_SUCCESS_TAG)
    if tag_idx == -1:
        return 0.0

    ans = extract_boxed_content(predict_str, start_after=tag_idx + len(TOOL_SUCCESS_TAG))
    return 1.0 if ans != "None" else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    tag_idx = predict_str.rfind(TOOL_SUCCESS_TAG)
    if tag_idx == -1:
        return 0.0

    if use_boxed:
        answer = extract_boxed_content(predict_str, start_after=tag_idx + len(TOOL_SUCCESS_TAG))
        if answer == "None":
            return 0.0
    else:
        answer = predict_str[tag_idx + len(TOOL_SUCCESS_TAG) :]

    return 1.0 if is_consistent(answer, ground_truth) else 0.0


def tool_reward(predict_str: str) -> float:
    matches = re.findall(re.escape(TOOL_SUCCESS_TAG), predict_str)
    if len(matches) == 1:
        return 1.0
    elif len(matches) > 1:
        return 0.5
    else:
        return 0.0


def compute_score(
    predict_str: str,
    ground_truth: str,
    use_boxed: bool = True,
    format_score: float = 0.1,
    tool_score: float = 0.1
) -> float:
    
    # breakpoint()
    tool_r = tool_reward(predict_str)
    acc_r = acc_reward(predict_str, ground_truth, use_boxed)
    format_r = format_reward(predict_str)

    # 沿用你的策略：若工具使用低于 0.5，则不计准确与格式（保证“答案必须在成功使用工具后”的语义）
    if tool_r < 0.5:
        acc_r = 0.0
        format_r = 0.0

    return (1.0 - format_score - tool_score) * acc_r + format_score * format_r + tool_score * tool_r
