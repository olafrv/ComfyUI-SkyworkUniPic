#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
from mmengine.config import Config
import torch
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange

from src.builder import BUILDER
from src.datasets.utils import crop2square


def preprocess_image(image: Image.Image, image_size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Process PIL image to normalized tensor [1,C,H,W]."""
    img = crop2square(image)
    img = img.resize((image_size, image_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = 2 * arr - 1
    tensor = torch.from_numpy(arr).to(dtype=dtype)
    return rearrange(tensor, "h w c -> 1 c h w")


class EditInferencer:
    def __init__(self, config_path: str, checkpoint_path: str, image_size: int):
        # 1) Build model
        self.cfg = Config.fromfile(config_path)
        self.model = BUILDER.build(self.cfg.model).eval().cuda().to(torch.bfloat16)

  
        # 2) Load model weights
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        # print(f"Unexpected parameters: {unexpected}")
        # print(f"Missed parameters: {missing}")

        self.image_size = image_size

        special_tokens_dict = {'additional_special_tokens': ["<image>", ]}
        num_added_toks = self.model.tokenizer.add_special_tokens(special_tokens_dict)

        self.image_token_idx = self.model.tokenizer.encode("<image>", add_special_tokens=False)[-1]
        print(f"Image token: {self.model.tokenizer.decode(self.image_token_idx)}")

    def edit_image(
        self, 
        source_image: Image.Image, 
        prompt: str, 
        num_iter: int = 48, 
        cfg: float = 3.0,
        cfg_prompt: str = "Repeat this image.",
        cfg_schedule: str = "constant",
        temperature: float = 0.85,
        grid_size: int = 1
    ) -> Image.Image:
        """Edit single image based on prompt."""
        # 1) Preprocess source image
        img_tensor = preprocess_image(
            source_image, 
            self.image_size,
            dtype=self.model.dtype
        ).to(self.model.device)
        
        # 2) Encode image and extract features
        with torch.no_grad():
            x_enc = self.model.encode(img_tensor)
            x_con, z_enc = self.model.extract_visual_feature(x_enc)
        
        # 3) Prepare text prompts
        m = n = self.image_size // 16
        image_length = m * n + 64
        
        if hasattr(self.cfg.model, 'prompt_template'):
            prompt_str = self.cfg.model.prompt_template['INSTRUCTION'].format(
                input="<image>\n" + prompt.strip()
            )
            cfg_prompt_str = self.cfg.model.prompt_template['INSTRUCTION'].format(
                input="<image>\n" + cfg_prompt.strip()
            )
        else:
            prompt_str = f"<image>\n{prompt.strip()}"
            cfg_prompt_str = f"<image>\n{cfg_prompt.strip()}"
        
        # Replace <image> token with multiple tokens
        prompt_str = prompt_str.replace('<image>', '<image>' * image_length)
        cfg_prompt_str = cfg_prompt_str.replace('<image>', '<image>' * image_length)
        
        # 4) Tokenize and prepare inputs
        input_ids = self.model.tokenizer.encode(
            prompt_str, add_special_tokens=True, return_tensors='pt')[0].cuda()
        
        if cfg != 1.0:
            null_input_ids = self.model.tokenizer.encode(
                cfg_prompt_str, add_special_tokens=True, return_tensors='pt')[0].cuda()
            attention_mask = pad_sequence(
                [torch.ones_like(input_ids), torch.ones_like(null_input_ids)],
                batch_first=True, padding_value=0).to(torch.bool)
            input_ids = pad_sequence(
                [input_ids, null_input_ids],
                batch_first=True, padding_value=self.model.tokenizer.eos_token_id)
        else:
            input_ids = input_ids[None]
            attention_mask = torch.ones_like(input_ids).to(torch.bool)
        
        # 5) Prepare embeddings
        if cfg != 1.0:
            z_enc = torch.cat([z_enc, z_enc], dim=0)
            x_con = torch.cat([x_con, x_con], dim=0)
        
        inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.model.llm.config.hidden_size)
        inputs_embeds[input_ids == self.image_token_idx] = z_enc.flatten(0, 1)
        inputs_embeds[input_ids != self.image_token_idx] = self.model.llm.get_input_embeddings()(
            input_ids[input_ids != self.image_token_idx]
        )
        
        # 6) Repeat for grid sampling
        bsz = grid_size ** 2
        x_con = torch.cat([x_con] * bsz)
        if cfg != 1.0:
            inputs_embeds = torch.cat([
                inputs_embeds[:1].expand(bsz, -1, -1),
                inputs_embeds[1:].expand(bsz, -1, -1),
            ])
            attention_mask = torch.cat([
                attention_mask[:1].expand(bsz, -1),
                attention_mask[1:].expand(bsz, -1),
            ])
        else:
            inputs_embeds = inputs_embeds.expand(bsz, -1, -1)
            attention_mask = attention_mask.expand(bsz, -1)
        
        # 7) Sampling
        samples = self.model.sample(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_iter=num_iter,
            cfg=cfg,
            cfg_schedule=cfg_schedule,
            temperature=temperature,
            progress=False,
            image_shape=(m, n),
            x_con=x_con
        )
        
        # 8) Convert to PIL Image
        samples = rearrange(samples, '(m n) c h w -> (m h) (n w) c', m=grid_size, n=grid_size)
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        out = samples.to("cpu", torch.uint8).numpy()
        return Image.fromarray(out)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


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


class LoadSkyworkUniPicImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "data/sample.png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "Skywork-UniPic"

    def load_image(self, image_path):
        image = image_path
        
        return (image,)


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


class Text2Image:
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

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "generate"
    CATEGORY = "Skywork-UniPic"

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


class SaveSkyworkUniPicImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "output.jpg"}),
                "samples": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save"
    CATEGORY = "Skywork-UniPic"

    def save(self, image_path, samples):
        Image.fromarray(samples).save(image_path)
        
        return ()


class ImageEditing:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("CONFIG",),
                "checkpoint": ("CHECKPOINT",),
                "image": ("IMAGE",),
                "prompt": ("PROMPT",),
                "cfg_prompt": ("STRING", {"default": "Repeat this image."}),
                "cfg": ("FLOAT", {"default": 3.0}),
                "temperature": ("FLOAT", {"default": 1.0}),
                "cfg_schedule": ("STRING", {"default": "constant"}),
                "num_iter": ("INT", {"default": 32}),
                "grid_size": ("INT", {"default": 1}),
                "image_size": ("INT", {"default": 1024}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("out_img",)
    FUNCTION = "generate"
    CATEGORY = "Skywork-UniPic"

    def generate(self, config, checkpoint, image, prompt, cfg_prompt, cfg, temperature, cfg_schedule, num_iter, grid_size, image_size):
        
        # Initialize inferencer on this GPU
        inferencer = EditInferencer(config_path = config, checkpoint_path = checkpoint, image_size =image_size)
        
        src_img = Image.open(image)
        out_img = inferencer.edit_image(
            source_image=src_img,
            prompt=prompt,
            num_iter=num_iter,
            cfg=cfg,
            cfg_prompt=cfg_prompt,
            cfg_schedule=cfg_schedule,
            temperature=temperature,
            grid_size=grid_size
        )

        return (out_img,)


class SaveSkyworkUniPicEditImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "output.jpg"}),
                "out_img": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save"
    CATEGORY = "Skywork-UniPic"

    def save(self, image_path, out_img):
        out_img.save(image_path)
        
        return ()


class Image2Text:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("CONFIG",),
                "checkpoint": ("CHECKPOINT",),
                "image": ("IMAGE",),
                "prompt": ("PROMPT",),
                "image_size": ("INT", {"default": 1024}),
            }
        }

    RETURN_TYPES = ("TEXT",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "Skywork-UniPic"

    def generate(self, config, checkpoint, image，prompt, image_size):
        
        config = Config.fromfile(config)
        model = BUILDER.build(config.model).eval().cuda()
        model = model.to(model.dtype)
        
        if checkpoint is not None:
            print(f"Load checkpoint: {checkpoint}", flush=True)
            checkpoint = torch.load(checkpoint)
            info = model.load_state_dict(checkpoint, strict=False)
    
        special_tokens_dict = {'additional_special_tokens': ["<image>", ]}
        num_added_toks = model.tokenizer.add_special_tokens(special_tokens_dict)
        # assert num_added_toks == 1
    
        image_token_idx = model.tokenizer.encode("<image>", add_special_tokens=False)[-1]
        print(f"Image token: {model.tokenizer.decode(image_token_idx)}")
    
        image = Image.open(image).convert('RGB')
    
        image = expand2square(
            image, (127, 127, 127))
        image = image.resize(size=(image_size, image_size))
        image = torch.from_numpy(np.array(image)).to(dtype=model.dtype, device=model.device)
        image = rearrange(image, 'h w c -> c h w')[None]
        image = 2 * (image / 255) - 1
    
        prompt = model.prompt_template['INSTRUCTION'].format(input="<image>\n" + prompt)
        assert '<image>' in prompt
        image_length = (image_size // 16) ** 2 + 64
        prompt = prompt.replace('<image>', '<image>'*image_length)
        input_ids = model.tokenizer.encode(
            prompt, add_special_tokens=True, return_tensors='pt').cuda()
        
        with torch.no_grad():
            _, z_enc = model.extract_visual_feature(model.encode(image))
        inputs_embeds = z_enc.new_zeros(*input_ids.shape, model.llm.config.hidden_size)
        inputs_embeds[input_ids == image_token_idx] = z_enc.flatten(0, 1)
        inputs_embeds[input_ids != image_token_idx] = model.llm.get_input_embeddings()(
            input_ids[input_ids != image_token_idx]
        )
        
        with torch.no_grad():
            output = model.llm.generate(inputs_embeds=inputs_embeds,
                                        use_cache=True,
                                        do_sample=False,
                                        max_new_tokens=1024,
                                        eos_token_id=model.tokenizer.eos_token_id,
                                        pad_token_id=model.tokenizer.pad_token_id
                                        if model.tokenizer.pad_token_id is not None else
                                        model.tokenizer.eos_token_id
                                        )
        print(model.tokenizer.decode(output[0]))

        text = model.tokenizer.decode(output[0])
        
        return (text,)

