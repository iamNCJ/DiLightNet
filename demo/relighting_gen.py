import imageio
import numpy as np
import torch
from diffusers import UniPCMultistepScheduler, StableDiffusionControlNetPipeline, StableDiffusionInpaintPipeline
from diffusers.utils import get_class_from_dynamic_module

from tqdm import tqdm

device = torch.device('cpu')
dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device('cuda')
    dtype = torch.float16

NeuralTextureControlNetModel = get_class_from_dynamic_module(
    "dilightnet/model_helpers",
    "neuraltexture_controlnet.py",
    "NeuralTextureControlNetModel"
)
controlnet = NeuralTextureControlNetModel.from_pretrained(
    "DiLightNet/DiLightNet",
    torch_dtype=dtype,
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    controlnet=controlnet,
    torch_dtype=dtype
).to(device)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=dtype
).to(device)
inpainting_pipe.set_progress_bar_config(disable=True)


def relighting_gen(masked_ref_img, mask, cond_path, frames, prompt, steps, seed, cfg, num_imgs_per_prompt=1, inpaint=False):
    mask = mask[..., :1] / 255.
    for i in tqdm(range(frames)):
        source_image = masked_ref_img[..., :3] / 255.

        hint_types = ['diffuse', 'ggx0.05', 'ggx0.13', 'ggx0.34']
        images = [mask, source_image]
        for hint_type in hint_types:
            image_path = f'{cond_path}/hint{i:02d}_{hint_type}.png'
            image = imageio.v3.imread(image_path) / 255.
            if image.shape[-1] == 4:  # Check if the image has an alpha channel
                image = image[..., :3] * image[..., 3:]  # Premultiply RGB by Alpha
            images.append(image)

        hint = np.concatenate(images, axis=2).astype(np.float32)[None]
        hint = torch.from_numpy(hint).to(dtype).permute(0, 3, 1, 2).to(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        images = pipe(
            prompt, num_inference_steps=steps, generator=generator, image=hint, num_images_per_prompt=num_imgs_per_prompt, guidance_scale=cfg, output_type='np',
        ).images  # [N, H, W, C]
        if inpaint:
            mask_image = (1. - mask)[None]
            images = inpainting_pipe(prompt=prompt, image=images, mask_image=mask_image, generator=generator, output_type='np', cfg=3.0, strength=1.0).images
        for idx in range(num_imgs_per_prompt):
            imageio.imwrite(f'{cond_path}/relighting{i:02d}_{idx}.png', (images[idx] * 255).clip(0, 255).astype(np.uint8))
