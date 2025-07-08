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

import json
import re
from typing import Any, Dict, List

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:

    # 1. 闲聊/回复格式奖励
    pattern = r'<think>\s*(.*?)\s*</think>\s*<answer>\s*(.*?)\s*</answer>\s*<next_question>\s*(.*?)\s*</next_question>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        think = match.group(1)
        if not think:
            return 0.5
        
        if len(think) > 100 or len(think) < 10:
            return 0.5
        
        return 1.0

    # 2. 检索步骤1格式奖励
    pattern = r'<think>\s*(.*?)\s*</think>\s*<search_query>\s*(.*?)\s*</search_query>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        try:
            think = match.group(1)
            if not think:
                return 0.2
            
            if len(think) > 100 or len(think) < 10:
                return 0.5
            
            data = json.loads(match.group(2))

            if not isinstance(data, dict):
                return 0.0
            
            if not 'query' in data:
                return 0.0
            
            return 1.0
        except:
            return 0.0
    return 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    # 先确保轨迹是正确的，也就是直接回复还是检索
    # TODO: 灵感问的奖励，引用的奖励，多轮检索的奖励
    if ground_truth == "检索":
        pattern = r'<think>\s*(.*?)\s*</think>\s*<search_query>\s*(.*?)\s*</search_query>'
        if re.search(pattern, response, re.DOTALL):
            return 1.0
        return 0.0
    elif ground_truth == "回答":
        pattern = r'<think>\s*(.*?)\s*</think>\s*<answer>\s*(.*?)\s*</answer>'
        if re.search(pattern, response, re.DOTALL):
            return 1.0
        return 0.0
    else:
        print(f"Unknown ground truth: {ground_truth}")
        return 0.0


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
