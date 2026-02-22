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

from rllm.rewards.rl_reward import rllm_reward_fn


def reward_func(data_source, solution_str, ground_truth, extra_info=None):
    if data_source in ["Maxwell-Jia/AIME_2024", "opencompass/cnmo2024_en", "opencompass/cnmo2024_zh"]:
        from recipe.r1.tasks import math

        return math.compute_score(solution_str, ground_truth)
    elif data_source == "Idavidrein/gpqa":
        from recipe.r1.tasks import gpqa

        return gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["livecodebench/code_generation_lite", "livecodebench/code_generation"]:
        from recipe.r1.tasks import livecodebench

        return livecodebench.compute_score(solution_str, ground_truth)

    elif data_source in ["aime2024", "aime2025", "amc23", "minerva", "olympiadbench", "math500"]: # 
        
        from recipe.r1.tasks import math
        
        res = math.compute_score(solution_str, ground_truth)
        return res

        # import pdb; pdb.set_trace()
        # if isinstance(solution_str, (list, dict)) or isinstance(ground_truth, (list, dict)):
        #     print(f"[DEBUG] Unexpected input type for math_dapo.compute_score: "
        #         f"solution_str={type(solution_str).__name__}, ground_truth={type(ground_truth).__name__}")
        #     print(f"[DEBUG] solution_str(head): {_head_str(solution_str)}")
        #     print(f"[DEBUG] ground_truth(head): {_head_str(ground_truth)}")
        #     import pdb; pdb.set_trace()

        # from recipe.r1.tasks import math_dapo
        # res = math_dapo.compute_score(solution_str, ground_truth)
        # return res

        # from recipe.entropy_math import compute_score 
        # res = compute_score(solution_str, ground_truth)
        # return res

    # elif data_source in ["math500"]:
    #     from recipe.r1.tasks import math_dapo
    #     res = math_dapo.compute_score(solution_str, ground_truth)

    elif data_source in ["livecodebench", "taco", "humanevalplus"]:        
        res = rllm_reward_fn(data_source, solution_str, ground_truth)

    else:
        raise NotImplementedError
