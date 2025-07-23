# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import pandas as pd

import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4

import re

# from verl.utils.reward_score import gsm8k
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

from verl.utils.dataset.vision_utils import process_image

from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# example tool call process in vlm outputs:

# <tool_call>{"name": "crop_image", "arguments":{"index": -1, "coordinates": "[100, 200, 120, 230]"}} </tool_call>


class Cxr1kTool(BaseTool):
    """A demo tool for calculating the reward of cxr1k.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema, full_parquet_path="./cxr_mini_crop/full.parquet"):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.full_info = pd.read_parquet(full_parquet_path)


    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        
        # breakpoint()

        basic_instructions = """format: <think>'your reason'</think>\\boxed{'your answer'}. 
To note, 'your reason' is your thinking steps for diagnosis.
'your answer' is only 'yes' or 'no', means whether the disease exists."""

        try:
            index = parameters.get("index", "")
            if not isinstance(index, int):
                index = int(index)
            
            current_info = self.full_info[self.full_info['seed'] == index]
            # img_256_path = current_info["img_256_path"].iloc[0]
            disease = current_info["disease"].iloc[0]
        except Exception as e:
            return {"image": [], "text": 
                    "You failed to correctly use the tool without correct image index repetition. Stop Now."}, 0.0, {}
        try:
            img_original_path = current_info["img_original_path"].iloc[0]

            coordinates = parameters.get("coordinates", None)

            # m = re.search(r"[(\d+),(\d+),(\d+),(\d+)]", coordinates)

            # x1, y1, x2, y2 = map(int, m.groups())

            crop_coords = [int(coord) for coord in coordinates] # tolerate more cases

            orig = Image.open(img_original_path).convert("RGB")
            # img_256 = Image.open(img_256_path).convert("RGB")
            fx, fy, fx2, fy2 = (
                crop_coords[i] * orig.size[i % 2] // 256
                for i in range(4)
            )
            crop_full = orig.crop((fx, fy, fx2, fy2))
            # resize cropped img to 256
            crop_256 = process_image(crop_full.resize((256, 256), resample=Image.LANCZOS))
        except Exception as e:
            return {"image": [], "text": 
                    "You failed to correctly use the tool without correct image index repetition. Stop Now."}, 0.0, {}
            # return {"image": [], "text": 
            #         "You failed to correctly use the tool. But forget about this, just use the original full view image, start your thinking then answer." + basic_instructions
            #         + f"Now, think then answer: based on both images, does {disease} exists in this patient, as shown in the X-ray?"}, 0.0, {}
        return {"image": [crop_256, ],
                 "text": 
                 """Here are the tool_call results. You successfully used the crop tool! Now <image> is the cropped region, start [stage 2] now.
Based on the full view image and cropped region image with a clearer view, start your thinking then answer.""" + basic_instructions
                 + f"Now, think then answer: based on both images, does {disease} exists in this patient, as shown in the X-ray?"
                }, 0.0, {}


    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
