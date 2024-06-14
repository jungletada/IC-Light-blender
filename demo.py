import os
import os.path as osp

import math
import numpy as np
import torch
import safetensors.torch as sf

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file
import warnings
warnings.filterwarnings('ignore')

import utils
from utils import pytorch2numpy, numpy2pytorch


sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# Change UNet
with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward

def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

unet.forward = hooked_unet_forward

# Load
model_path = './models/iclight_sd15_fc.safetensors'

if not os.path.exists(model_path):
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/blob/main/iclight_sd15_fc.safetensors', dst=model_path)

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device
device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP
unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers
ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Pipelines
t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = utils.cv2_resize_img(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    alpha[alpha < 0.97] = 0
    alpha[alpha >= 0.97] = 1
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


@torch.inference_mode()
def process(input_fg, mask, prompt, num_samples, seed, steps, a_prompt, n_prompt, 
            cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    fg = utils.cv2_resize_img_aspect(input_fg)
    mask = utils.cv2_resize_img_aspect(mask)
    image_width, image_height = fg.shape[:2]
    
    bg_source = BGSource(bg_source)
    input_bg = None
    if bg_source == BGSource.NONE:
        pass
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(255, 0, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(0, 255, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(255, 0, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(0, 255, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise 'Wrong initial latent!'

    rng = torch.Generator(device=device).manual_seed(int(seed))
    
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor # (B, 4, H//8, W//8)
    
    conds, unconds = encode_prompt_pair(
        positive_prompt=prompt + ', ' + a_prompt, 
        negative_prompt=n_prompt) # (B, 77, 768)
    
    if input_bg is None:
        latents = t2i_pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor
        
    else:
        bg = utils.cv2_resize_img(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        latents = i2i_pipe(
            image=bg_latent,
            strength=lowres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / lowres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type='latent',
            guidance_scale=cfg,
            cross_attention_kwargs={'concat_conds': concat_conds},
        ).images.to(vae.dtype) / vae.config.scaling_factor
    
    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [utils.cv2_resize_img(
        img=p,
        new_w=int(round(image_width * highres_scale / 64.0) * 64),
        new_h=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]
    
    for i, p in enumerate(pixels):
        utils.cv2_save_rgb(f'results/bg-{i}.png', p)
    # pixels here are the (background) and (foreground lighting) generated by SD
    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    
    # Encode pixels by vae
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8
    fg = utils.cv2_resize_img(fg, image_width, image_height)
    mask = utils.cv2_resize_img(mask, image_width, image_height)

    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    
    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels, quant=False)

    return pixels, fg, mask


@torch.inference_mode()
def process_relight(input_fg, input_bg, prompt, num_samples, seed, steps, a_prompt, n_prompt, 
                    cfg, highres_scale, highres_denoise, bg_source, blend_threshold):
    input_fg, mask = run_rmbg(input_fg)
    results, fg, mask, bg = process(
        input_fg, input_bg, mask, prompt, num_samples, seed, steps, a_prompt, n_prompt, 
        cfg, highres_scale, highres_denoise, bg_source)
    results = [(x * 255.0).clip(0, 255).astype(np.uint8) for x in results]
    
    mask = mask[...,np.newaxis].repeat(3, axis=2)
    results = utils.blend_ic_light_bg(
        mask, fg, bg, results, threshold=blend_threshold)
    
    return results


quick_prompts = [
    'indoor',
    'outdoor',
    'sunshine from window',
    'neon light, city',
    'sunset over sea',
    'golden time',
    'sci-fi RGB glowing, cyberpunk',
    'natural lighting',
    'warm atmosphere, at home, bedroom',
    'magic lit',
    'evil, gothic, Yharnam',
    'light and shadow',
    'shadow from window',
    'soft studio lighting',
    'home atmosphere, cozy bedroom illumination',
    'neon, Wong Kar-wai, warm'
]

quick_subjects = [
    'beautiful woman, detailed face',
    'handsome man, detailed face',
]

class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"


if __name__ == '__main__':
    path = 'results'
    input_fg = utils.cv2_load_rgb(osp.join(path, 'lady-photo-5.jpeg')) # 前景图片
    prompt = 'neon light, city' # 基本prompt
    num_samples = 2 # 生成图片数量
    seed = 13233    # 随机种子
    steps = 20      # 去噪步数
    a_prompt = 'best quality' # 正面prompt
    n_prompt = 'lowres, bad anatomy, bad hands, cropped, worst quality' # 负面prompt
    cfg = 7.0       # classfier guidance
    highres_scale = 1.0     # 放到倍数
    highres_denoise = 0.5   # 第一阶段去噪强度
    lowres_denoise = 0.9    # 第二阶段去噪强度
    bg_source = BGSource.NONE  # 参考背景图像光照
    blend_threshold = 0.16     # 前景光照混合权重阈值

    # 1 显著性分割, 这一步可以替换为SAM分割
    input_fg, mask = run_rmbg(img=input_fg)
    # 1.1 考虑采用开运算或者腐蚀操作处理mask
    # mask = utils.cv2_morphologyEx(mask, kernel_size=(5, 5))
    mask = utils.cv2_erode_image(mask, kernel_size=(1, 1))
    
    # 2 图像生成
    results, output_fg, mask = process(
        input_fg=input_fg,
        mask=mask,
        prompt=prompt,
        num_samples=num_samples,
        seed=seed,
        steps=steps,
        a_prompt=a_prompt,
        n_prompt=n_prompt,
        cfg=cfg,
        highres_scale=highres_scale,
        highres_denoise=highres_denoise,
        lowres_denoise=lowres_denoise,
        bg_source=bg_source)
        
    # 2.1 保证输出的取值在(0, 255)
    results = [(x * 255.0).clip(0, 255).astype(np.uint8) for x in results]
    # 2.2 mask广播为三通道方便乘法
    mask = mask[...,np.newaxis].repeat(3, axis=2)
    for i, ic_res in enumerate(results):
        print(f'saving image {i}...')
        utils.cv2_save_rgb(osp.join(path, f'lady-photo-5-ic-{i}.jpeg'), ic_res)
    # 3 生成图像与原图像的加权求和
    result_gallery = utils.blend_ic_light(
        resized_fg=output_fg,  
        resized_mask=mask,
        ic_results=results, 
        blend_value=blend_threshold)

    for i, ic_res in enumerate(result_gallery):
        print(f'saving image {i}...')
        utils.cv2_save_rgb(osp.join(path, f'lady-photo-5-ic-b{i}.jpeg'), ic_res)
