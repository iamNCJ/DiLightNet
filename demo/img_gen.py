import gradio as gr
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


model_id = "stabilityai/stable-diffusion-2-1"

device = torch.device('cpu')
dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device('cuda')
    dtype = torch.float16

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)


def img_gen(prompt, seed, steps, cfg, down_from_768=False, progress=gr.Progress(track_tqdm=True)):
    generator = torch.Generator(device=device).manual_seed(int(seed))
    hw = 512 if not down_from_768 else 768
    image = pipe(prompt, generator=generator, num_inference_steps=int(steps), guidance_scale=cfg, output_type='np', height=hw, width=hw).images[0]
    if down_from_768:
        image = F.interpolate(torch.from_numpy(image)[None].permute(0, 3, 1, 2), size=(512, 512), mode='bilinear', align_corners=False, antialias=True).permute(0, 2, 3, 1)[0].cpu().numpy()
    return image
