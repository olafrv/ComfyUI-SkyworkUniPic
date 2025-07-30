import torch
from src.builder import BUILDER
from PIL import Image
from mmengine.config import Config
import argparse
from einops import rearrange


class LoadSkyworkUniPicConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config_path": ("STRING", {"default": "configs/models/qwen2_5_1_5b_kl16_mar_h.py"}),
            }
        }

    RETURN_TYPES = ("CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "load_config"
    CATEGORY = "Skywork-UniPic"

    def load_config(self, config_path):
        config = config_path
        
        return (config,)


class LoadSkyworkUniPicCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": ("STRING", {"default": "checkpoint/pytorch_model.bin"}),
            }
        }

    RETURN_TYPES = ("CHECKPOINT",)
    RETURN_NAMES = ("checkpoint",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "Skywork-UniPic"

    def load_checkpoint(self, checkpoint_path):
        checkpoint = checkpoint_path
        
        return (checkpoint,)


class LoadSkyworkUniPicPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "A glossy-coated golden retriever stands on the park lawn beside a life-sized penguin statue.",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "Skywork-UniPic"

    def load_prompt(self, text):
        prompt = text
        
        return (prompt,)


class SkyworkUniPic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("CONFIG",),
                "checkpoint": ("CHECKPOINT",),
                "prompt": ("PROMPT",),
                "cfg_prompt": ("STRING", {"default": "Generate an image."}),
                "cfg": ("FLOAT", {"default": 3.0}),
                "temperature": ("FLOAT", {"default": 1.0}),
                "cfg_schedule": ("STRING", {"default": "constant"}),
                "num_iter": ("INT", {"default": 48}),
                "grid_size": ("INT", {"default": 2}),
                "image_size": ("INT", {"default": 1024}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("output",)
    FUNCTION = "generate"
    CATEGORY = "Higgs Audio"

    def generate(self, config, checkpoint, prompt, cfg_prompt, cfg, temperature, cfg_schedule, num_iter, grid_size, image_size):
        
        config = Config.fromfile(config)
        model = BUILDER.build(config.model).eval().cuda()
        model = model.to(model.dtype)
        
        checkpoint = torch.load(checkpoint)
        info = model.load_state_dict(checkpoint, strict=False)
        
        prompt = f"Generate an image: {prompt}"
        print(prompt, flush=True)
        
        class_info = model.prepare_text_conditions(prompt, cfg_prompt)
    
        input_ids = class_info['input_ids']
        attention_mask = class_info['attention_mask']
    
        assert len(input_ids) == 2    # the last one is unconditional prompt
        if cfg == 1.0:
            input_ids = input_ids[:1]
            attention_mask = attention_mask[:1]
    
        # repeat
        bsz = grid_size ** 2
        if cfg != 1.0:
            input_ids = torch.cat([
                input_ids[:1].expand(bsz, -1),
                input_ids[1:].expand(bsz, -1),
            ])
            attention_mask = torch.cat([
                attention_mask[:1].expand(bsz, -1),
                attention_mask[1:].expand(bsz, -1),
            ])
        else:
            input_ids = input_ids.expand(bsz, -1)
            attention_mask = attention_mask.expand(bsz, -1)
     
        m = n = image_size // 16
    
        samples = model.sample(input_ids=input_ids, attention_mask=attention_mask,
                               num_iter=num_iter, cfg=cfg, cfg_schedule=cfg_schedule,
                               temperature=temperature, progress=True, image_shape=(m, n))
        
        samples = rearrange(samples, '(m n) c h w -> (m h) (n w) c', m=grid_size, n=grid_size)
        samples = torch.clamp(
            127.5 * samples + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
        
        return (samples,)


