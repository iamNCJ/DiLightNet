import imageio
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import get_class_from_dynamic_module


# load pipelines
device = 'cuda'
NeuralTextureControlNetModel = get_class_from_dynamic_module(
    "dilightnet/model_helpers",
    "neuraltexture_controlnet.py",
    "NeuralTextureControlNetModel"
)
neuraltexture_controlnet = NeuralTextureControlNetModel.from_pretrained("DiLightNet/DiLightNet")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", controlnet=neuraltexture_controlnet,
).to(device)

# load cond images
cond_img_path = 'examples/input/futuristic_soldier'
mask = imageio.v3.imread(cond_img_path + '_mask.png')[..., :1] / 255.
source_image = imageio.v3.imread(cond_img_path + '.png')[..., :3] / 255.
hint_types = ['diffuse', 'ggx0.05', 'ggx0.13', 'ggx0.34']
images = [mask, source_image]
for hint_type in hint_types:
    image_path = f'{cond_img_path}_{hint_type}.png'
    image = imageio.v3.imread(image_path) / 255.
    if image.shape[-1] == 4:  # Check if the image has an alpha channel
        image = image[..., :3] * image[..., 3:]  # Premultiply RGB by Alpha
    images.append(image)
hint = np.concatenate(images, axis=2).astype(np.float32)[None]
hint = torch.from_numpy(hint).to(torch.float32).permute(0, 3, 1, 2).to(device)

# run pipeline
image = pipe("futuristic soldier with advanced armor weaponry and helmet", image=hint).images[0]
image.save('output.png')
