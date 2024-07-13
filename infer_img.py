import os
from dataclasses import dataclass
from typing import Optional

import imageio
import numpy as np
import cv2
import simple_parsing


@dataclass
class Args:
    prov_img: str  # Path to the provisional image
    prompt: str = ""  # Prompt for the generated images
    num_imgs_per_prompt: int = 4  # Number of images to generate per prompt
    out_vid: Optional[str] = None # Path to the output video, defaults to the input image path

    seed: int = 3407  # Seed for the generation
    steps: int = 20  # Number of steps for the diffusion process
    cfg: float = 3.0  # CFG for the diffusion process

    fov: Optional[float] = None  # Field of view for the mesh reconstruction, none for auto estimation from the image

    mask_path: Optional[str] = None  # Path to the mask for the image
    use_sam: bool = True  # Use SAM for background removal
    mask_threshold: float = 25.  # Mask threshold for foreground object extraction

    pl_pos_r: float = 5.  # Rotation radius of the point light
    pl_pos_h: float = 3.  # Height of the point light
    power: float = 1200.  # Power of the point light
    inpaint: bool = False  # Inpaint the background of generated point light images
    env_map: Optional[str] = None  # Environment map for the rendering, defaults to None (white point light)
    frames: int = 120  # Number of frames for lighting controlled video
    use_gpu_for_rendering: bool = True  # Use GPU for radiance hints rendering

    cache_radiance_hints: bool = True  # Cache the radiance hints for the video
    radiance_hints_path: Optional[str] = None  # pre-rendered radiance hint path


if __name__ == '__main__':
    args = simple_parsing.parse(Args)

    from demo.mesh_recon import mesh_reconstruction
    from demo.relighting_gen import relighting_gen
    from demo.render_hints import render_hint_images, render_bg_images
    from demo.rm_bg import rm_bg

    # Load input image and generate/load mask
    input_image = imageio.v3.imread(args.prov_img)
    input_image = cv2.resize(input_image, (512, 512))
    if args.mask_path:
        mask = imageio.v3.imread(args.mask_path)
        if mask.ndim == 3:
            mask = mask[..., -1]
        mask = cv2.resize(mask, (512, 512))
    else:
        _, mask = rm_bg(input_image, use_sam=args.use_sam)
    mask = mask[..., None].repeat(3, axis=-1)

    frames = args.frames
    if args.radiance_hints_path is not None:
        res_path = args.radiance_hints_path
        print(f'Using pre-rendered radiance hints in {res_path}')
        use_env_map = os.path.exists(f'{res_path}/bg00.png')
    else:
        # Render radiance hints
        pls = [(
            args.pl_pos_r * np.sin(frame / frames * np.pi * 2.),
            args.pl_pos_r * np.cos(frame / frames * np.pi * 2.),
            args.pl_pos_h
        ) for frame in range(frames)]

        # cache middle results
        prov_img_id = os.path.basename(args.prov_img).split(".")[0]
        lighting_id = f'env_map-{os.path.basename(args.env_map).split(".")[0] if args.env_map else f"pl-{args.pl_pos_r}-{args.pl_pos_h}-{args.power}"}'
        frame_num_id = f'frames-{frames}'
        output_folder = f'tmp/{prov_img_id}/{lighting_id}/{frame_num_id}'
        os.makedirs(output_folder, exist_ok=True)

        # use cache if possible
        use_env_map = args.env_map is not None
        render_radiance_hints = True
        render_env_bg = use_env_map
        if args.cache_radiance_hints:
            # check if the radiance hints are already rendered and full
            render_radiance_hints = False
            for i in range(frames):
                if not (os.path.exists(f'{output_folder}/hint{i:02d}_diffuse.png')
                        and os.path.exists(f'{output_folder}/hint{i:02d}_ggx0.05.png')
                        and os.path.exists(f'{output_folder}/hint{i:02d}_ggx0.13.png')
                        and os.path.exists(f'{output_folder}/hint{i:02d}_ggx0.34.png')):

                    render_radiance_hints = True
                    break
            # check if the radiance hints are already rendered and full
            if use_env_map:
                render_env_bg = False
                for i in range(frames):
                    if not os.path.exists(f'{output_folder}/bg{i:02d}.png'):
                        render_env_bg = True
                        break
        print(f"Rendering radiance hints: {render_radiance_hints}")
        print(f"Rendering env bg: {render_env_bg}")

        # Render hints if needed
        if render_radiance_hints or render_env_bg:
            # Mesh reconstruction and fov estimation for hints rendering
            fov = args.fov
            mesh, fov = mesh_reconstruction(input_image, mask, False, fov, args.mask_threshold)
            print(f"Mesh reconstructed with fov: {fov}")
        if render_radiance_hints:
            render_hint_images(mesh, fov, pls, args.power, env_map=args.env_map, output_folder=output_folder, use_gpu=args.use_gpu_for_rendering)
        if render_env_bg:
            render_bg_images(fov, pls, env_map=args.env_map, output_folder=output_folder, use_gpu=args.use_gpu_for_rendering)
        res_path = output_folder

    # Generate relighting images
    mask_for_bg = imageio.v3.imread(res_path + '/hint00_diffuse.png')[..., -1:] / 255.
    masked_image = (input_image.astype(np.float32) * mask_for_bg).clip(0, 255).astype(np.uint8)
    relighting_gen(
        masked_ref_img=masked_image,
        mask=mask,
        cond_path=res_path,
        frames=frames,
        prompt=args.prompt,
        steps=args.steps,
        seed=args.seed,
        cfg=args.cfg,
        num_imgs_per_prompt=args.num_imgs_per_prompt,
        inpaint=not use_env_map and args.inpaint,
    )

    # Assemble the video
    for idx in range(args.num_imgs_per_prompt):
        all_res = []
        for frame in range(frames):
            relit_img = imageio.v3.imread(res_path + f'/relighting{frame:02d}_{idx}.png')
            if use_env_map:
                bg = imageio.v3.imread(res_path + f'/bg{frame:02d}.png') / 255.
                relit_img = relit_img / 255.
                relit_img = relit_img * mask_for_bg + bg * (1. - mask_for_bg)
                relit_img = (relit_img * 255).clip(0, 255).astype(np.uint8)
            all_res.append(relit_img)
        all_res = np.stack(all_res, axis=0)
        out_vid = args.out_vid or args.prov_img
        out_vid = f'{os.path.splitext(out_vid)[0]}_{idx}.mp4'
        os.makedirs(os.path.dirname(out_vid), exist_ok=True)
        imageio.v3.imwrite(out_vid, all_res, fps=24, quality=9)
