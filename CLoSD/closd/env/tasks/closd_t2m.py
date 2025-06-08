# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from closd.env.tasks import closd_task
from isaacgym.torch_utils import *
from closd.utils.closd_util import STATES

### for custom t2m ###
from transformers import BertTokenizer
from pathlib import Path
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
import time 

class CLoSDT2M(closd_task.CLoSDTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        self.init_state = STATES.TEXT2MOTION
        self.hml_data_buf_size = max(self.fake_mdm_args.context_len, self.planning_horizon_20fps)
        self.hml_prefix_from_data = torch.zeros([self.num_envs, 263, 1, self.hml_data_buf_size], dtype=torch.float32, device=self.device)
        return
    # version 4 
    
    def update_mdm_conditions(self, env_ids):  
        super().update_mdm_conditions(env_ids)


        try:
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        except StopIteration:
            del self.mdm_data_iter
            self.mdm_data_iter = iter(self.mdm_data)
            gt_motion, model_kwargs = next(self.mdm_data_iter)

        # ===== [CLOSD MOD] Queue pop from prompt_queue.txt =====
        prompt_path = "/home/bong/CLoSD/closd/custom_t2m/prompt_queue.txt"
    
        if Path(prompt_path).exists():
            lines = Path(prompt_path).read_text().strip().splitlines()
            if len(lines) > 0:
                input_command = lines[0]
                remaining = "\n".join(lines[1:])
                Path(prompt_path).write_text(remaining)
            else:
                input_command = ""
        else:
            input_command = ""

        if input_command == "":
            input_command = "A person is standing still."
            
        # 수정: prompt 정보를 파일명 정할 때 쓸 수 있도록
        self.prompt_now = input_command
        
        print(f"[Prompt]: {input_command}")
        
        
        '''
        if Path(prompt_path).exists():
            
            # print("[CLOSD DEBUG] Reading prompt from queue file...")  
            lines = Path(prompt_path).read_text().strip().splitlines()
            if len(lines) > 0:
                input_command = lines[0]  # take first prompt
                remaining = "\n".join(lines[1:])
                Path(prompt_path).write_text(remaining)  # remove first line
            else:
                input_command = ""
        else:
            input_command = ""
            # print("[CLOSD DEBUG] Queue file not found. Using fallback.")  

        if input_command == "":
            input_command = "A person is standing still."
            
        print(f"[Prompt: {input_command}", end="", flush=True)
        '''
        
        # ===== [CLOSD MOD END] =====

        tokenized = tokenizer(
            input_command,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=20
        )
        tokens = tokenized["input_ids"][0]
        length = (tokens != tokenizer.pad_token_id).sum().item()
        
        for i in env_ids:
            self.hml_prompts[int(i)] = input_command
            self.hml_lengths[int(i)] = torch.tensor(length, device=self.device)
            self.hml_tokens[int(i)] = tokens.to(self.device)
            self.db_keys[int(i)] = model_kwargs['y']['db_key'][int(i)]

            # print(f"[Env {int(i)}] Prompt: {input_command}")
            
    # version 3 (read prompt from txt file)
    '''
    def update_mdm_conditions(self, env_ids):  
        super().update_mdm_conditions(env_ids)

        try:
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        except StopIteration:
            del self.mdm_data_iter
            self.mdm_data_iter = iter(self.mdm_data)
            gt_motion, model_kwargs = next(self.mdm_data_iter)

        # ===== [CLOSD MOD] Queue pop from prompt_queue.txt =====
        prompt_path = "/home/bong/CLoSD/closd/custom_t2m/prompt_queue.txt"
        if Path(prompt_path).exists():
            lines = Path(prompt_path).read_text().strip().splitlines()
            if len(lines) > 0:
                input_command = lines[0]  # take first prompt
                remaining = "\n".join(lines[1:])
                Path(prompt_path).write_text(remaining)  # remove first line
            else:
                input_command = ""
        else:
            input_command = ""

        if input_command == "":
            input_command = "A person is standing still."
        # ===== [CLOSD MOD END] =====

        tokenized = tokenizer(
            input_command,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=20
        )
        tokens = tokenized["input_ids"][0]
        length = (tokens != tokenizer.pad_token_id).sum().item()

        for i in env_ids:
            self.hml_prompts[int(i)] = input_command
            self.hml_lengths[int(i)] = torch.tensor(length, device=self.device)
            self.hml_tokens[int(i)] = tokens.to(self.device)
            self.db_keys[int(i)] = model_kwargs['y']['db_key'][int(i)]

            print(f"[Env {int(i)}] Prompt: {input_command}")
            
    '''


    # version 2 (read prompt from queue)
    '''
    def update_mdm_conditions(self, env_ids):  
        super().update_mdm_conditions(env_ids)

        try:
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        except StopIteration:
            del self.mdm_data_iter
            self.mdm_data_iter = iter(self.mdm_data)
            gt_motion, model_kwargs = next(self.mdm_data_iter)

        # ===== [CLOSD MOD] Get prompt from queue or fallback =====
        try:
            input_command = custom_prompt.get_nowait()
            print(f"[CLOSD DEBUG] Got prompt: {input_command}") #debug
        except:
            input_command = "A person is standing still."
            print("[CLOSD DEBUG] Queue empty. Using fallback.") #debug
        # ===== [CLOSD MOD END] =====


        tokenized = tokenizer(
            input_command,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=20
        )
        tokens = tokenized["input_ids"][0]
        length = (tokens != tokenizer.pad_token_id).sum().item()

        for i in env_ids:
            self.hml_prompts[int(i)] = input_command
            self.hml_lengths[int(i)] = torch.tensor(length, device=self.device)
            self.hml_tokens[int(i)] = tokens.to(self.device)
            self.db_keys[int(i)] = model_kwargs['y']['db_key'][int(i)]

            print(f"[Env {int(i)}] Prompt: {input_command}")
    '''

    # version 1 (get motion from input())
    '''
    def update_mdm_conditions(self, env_ids):  
        super().update_mdm_conditions(env_ids)

        try:
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        except StopIteration:
            del self.mdm_data_iter
            self.mdm_data_iter = iter(self.mdm_data)
            gt_motion, model_kwargs = next(self.mdm_data_iter)

        # ===== [CLOSD MOD] Prompt from input(), fallback to default =====
        input_command = input("\n[Enter motion prompt] (press Enter to stand still) > ").strip()
        if input_command == "":
            input_command = "A person is standing still."

        tokenized = tokenizer(
            input_command,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=20
        )
        tokens = tokenized["input_ids"][0]
        length = (tokens != tokenizer.pad_token_id).sum().item()
        # ===== [CLOSD MOD END] =====

        for i in env_ids:
            self.hml_prompts[int(i)] = input_command
            self.hml_lengths[int(i)] = torch.tensor(length, device=self.device)
            self.hml_tokens[int(i)] = tokens.to(self.device)
            self.db_keys[int(i)] = model_kwargs['y']['db_key'][int(i)]

            print(f"[Env {int(i)}] Prompt: {input_command}")
    '''

    # Original
    '''
    
    def update_mdm_conditions(self, env_ids):  
        super().update_mdm_conditions(env_ids)
        
        # updates prompts and lengths
        try:
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        except StopIteration:
            del self.mdm_data_iter
            self.mdm_data_iter = iter(self.mdm_data) # re-initialize
            gt_motion, model_kwargs = next(self.mdm_data_iter)
        for i in env_ids:
            self.hml_prompts[int(i)] = model_kwargs['y']['text'][int(i)]
            self.hml_lengths[int(i)] = model_kwargs['y']['lengths'][int(i)]  
            self.hml_tokens[int(i)] = model_kwargs['y']['tokens'][int(i)]  
            self.db_keys[int(i)] = model_kwargs['y']['db_key'][int(i)]  
        self.hml_prefix_from_data[env_ids] = gt_motion[..., :self.hml_data_buf_size].to(self.device)[env_ids]  # will be used by the first MDM iteration
        if self.cfg['env']['dip']['debug_hml']:
            print(f'in update_mdm_conditions: 1st 10 env_ids={env_ids[:10].cpu().numpy()}, prompts={self.hml_prompts[:2]}')
        return
    '''
    
    def get_cur_done(self):
        # Done signal is not in use for this task
        return torch.zeros([self.num_envs], device=self.device, dtype=bool)
    

