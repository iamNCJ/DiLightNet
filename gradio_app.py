import gradio as gr
import os
import imageio
import numpy as np
from einops import rearrange

from demo.img_gen import img_gen
from demo.mesh_recon import mesh_reconstruction
from demo.relighting_gen import relighting_gen
from demo.render_hints import render_hint_images_btn_func
from demo.rm_bg import rm_bg


with gr.Blocks(title="DiLightNet Demo") as demo:
    gr.Markdown("""# DiLightNet: Fine-grained Lighting Control for Diffusion-based Image Generation
                ## A demo for generating images under point/environmantal lighting using DiLightNet. For full usage (video generation & arbitary lighting condition), please refer to our [GitHub repository](https://github.com/iamNCJ/DiLightNet)""")

    with gr.Row():
        # 1. Reference Image Input / Generation
        with gr.Column(variant="panel"):
            gr.Markdown("## Step 1. Input or Generate Reference Image")
            input_image = gr.Image(height=512, width=512, label="Input Image", interactive=True)
            with gr.Accordion("Generate Image", open=False):
                with gr.Group():
                    prompt = gr.Textbox(value="", label="Prompt", lines=3, placeholder="Input prompt here")
                    with gr.Row():
                        seed = gr.Number(value=42, label="Seed", interactive=True)
                        steps = gr.Number(value=20, label="Steps", interactive=True)
                        cfg = gr.Number(value=7.5, label="CFG", interactive=True)
                        down_from_768 = gr.Checkbox(label="Downsample from 768", value=True)
                with gr.Row():
                    generate_btn = gr.Button(value="Generate")
                    generate_btn.click(fn=img_gen, inputs=[prompt, seed, steps, cfg, down_from_768], outputs=[input_image])
            gr.Examples(
                examples=[os.path.join("examples/provisional_img", i) for i in os.listdir("examples/provisional_img")],
                inputs=[input_image],
                examples_per_page=8,
            )

        # 2. Background Removal
        with gr.Column(variant="panel"):
            gr.Markdown("## Step 2. Remove Background")
            with gr.Tab("Masked Image"):
                masked_image = gr.Image(height=512, width=512, label="Masked Image", interactive=True)
            with gr.Tab("Mask"):
                mask = gr.Image(height=512, width=512, label="Mask", interactive=False)
            use_sam = gr.Checkbox(label="Use SAM for Refinement", value=False)
            rm_bg_btn = gr.Button(value="Remove Background")
            rm_bg_btn.click(fn=rm_bg, inputs=[input_image, use_sam], outputs=[masked_image, mask])

        # 3. Depth Estimation & Mesh Reconstruction
        with gr.Column(variant="panel"):
            gr.Markdown("## Step 3. Depth Estimation & Mesh Reconstruction")
            mesh = gr.Model3D(label="Mesh Reconstruction", clear_color=(1.0, 1.0, 1.0, 1.0), interactive=True)
            with gr.Column():
                with gr.Accordion("Options", open=False):
                    with gr.Group():
                        remove_edges = gr.Checkbox(label="Remove Occlusion Edges", value=False)
                        fov = gr.Number(value=55., label="FOV", interactive=False)
                        mask_threshold = gr.Slider(value=25., label="Mask Threshold", minimum=0., maximum=255., step=1.)
                depth_estimation_btn = gr.Button(value="Estimate Depth")
                def mesh_reconstruction_wrapper(image, mask, remove_edges, mask_threshold,
                                                progress=gr.Progress(track_tqdm=True)):
                    return mesh_reconstruction(image, mask, remove_edges, None, mask_threshold)
                depth_estimation_btn.click(
                    fn=mesh_reconstruction_wrapper,
                    inputs=[input_image, mask, remove_edges, mask_threshold],
                    outputs=[mesh, fov],
                )

    with gr.Row():
        with gr.Column(variant="panel"):
            gr.Markdown("## Step 4. Render Hints")
            hint_image = gr.Image(label="Hint Image", height=512, width=512)
            res_folder_path = gr.Textbox("", visible=False)
            is_env_lighting = gr.Checkbox(label="Use Environmental Lighting", value=True, interactive=False, visible=False)
            with gr.Tab("Environmental Lighting"):
                env_map_preview = gr.Image(label="Environment Map Preview", height=256, width=512, interactive=False, show_download_button=False)
                env_map_path = gr.Text(interactive=False, visible=False, value="examples/env_map/grace.exr")
                env_rotation = gr.Slider(value=0., label="Environment Rotation", minimum=0., maximum=360., step=0.5)
                env_examples = gr.Examples(
                    examples=[[os.path.join("examples/env_map_preview", i), os.path.join("examples/env_map", i).replace("png", "exr")] for i in os.listdir("examples/env_map_preview")],
                    inputs=[env_map_preview, env_map_path],
                    examples_per_page=20,
                )
                render_btn_env = gr.Button(value="Render Hints")

                def render_wrapper_env(mesh, fov, env_map_path, env_rotation, progress=gr.Progress(track_tqdm=True)):
                    env_map_path = os.path.abspath(env_map_path)
                    res_path = render_hint_images_btn_func(mesh, float(fov), [(0, 0, 0)], env_map=env_map_path, env_start_azi=env_rotation / 360.)
                    hint_files = [res_path + '/hint00' + mat for mat in ["_diffuse.png", "_ggx0.05.png", "_ggx0.13.png", "_ggx0.34.png"]]
                    hints = []
                    for hint_file in hint_files:
                        hint = imageio.v3.imread(hint_file)
                        hints.append(hint)
                    hints = rearrange(np.stack(hints), '(n1 n2) h w c -> (n1 h) (n2 w) c', n1=2, n2=2)
                    return hints, res_path, True
                render_btn_env.click(
                    fn=render_wrapper_env,
                    inputs=[mesh, fov, env_map_path, env_rotation],
                    outputs=[hint_image, res_folder_path, is_env_lighting]
                )

            with gr.Tab("Point Lighting"):
                pl_pos_x = gr.Slider(value=3., label="Point Light X", minimum=-5., maximum=5., step=0.01)
                pl_pos_y = gr.Slider(value=1., label="Point Light Y", minimum=-5., maximum=5., step=0.01)
                pl_pos_z = gr.Slider(value=3., label="Point Light Z", minimum=-5., maximum=5., step=0.01)
                power = gr.Slider(value=1000., label="Point Light Power", minimum=0., maximum=2000., step=1.)
                render_btn_pl = gr.Button(value="Render Hints")

                def render_wrapper_pl(mesh, fov, pl_pos_x, pl_pos_y, pl_pos_z, power,
                                progress=gr.Progress(track_tqdm=True)):
                    res_path = render_hint_images_btn_func(mesh, float(fov), [(pl_pos_x, pl_pos_y, pl_pos_z)], power)
                    hint_files = [res_path + '/hint00' + mat for mat in ["_diffuse.png", "_ggx0.05.png", "_ggx0.13.png", "_ggx0.34.png"]]
                    hints = []
                    for hint_file in hint_files:
                        hint = imageio.v3.imread(hint_file)
                        hints.append(hint)
                    hints = rearrange(np.stack(hints), '(n1 n2) h w c -> (n1 h) (n2 w) c', n1=2, n2=2)
                    return hints, res_path, False

                render_btn_pl.click(
                    fn=render_wrapper_pl,
                    inputs=[mesh, fov, pl_pos_x, pl_pos_y, pl_pos_z, power],
                    outputs=[hint_image, res_folder_path, is_env_lighting]
                )

        with gr.Column(variant="panel"):
            gr.Markdown("## Step 5. Control Lighting!")
            res_image = gr.Image(label="Result Image", height=512, width=512)
            with gr.Group():
                relighting_prompt = gr.Textbox(value="", label="Appearance Text Prompt", lines=3,
                                                placeholder="Input prompt here",
                                                interactive=True)
                # several example prompts
                with gr.Row():
                    metallic_prompt_btn = gr.Button(value="Metallic", size="sm")
                    specular_prompt_btn = gr.Button(value="Specular", size="sm")
                    very_specular_prompt_btn = gr.Button(value="Very Specular", size="sm")
                metallic_prompt_btn.click(fn=lambda x: x + " metallic", inputs=[relighting_prompt], outputs=[relighting_prompt])
                specular_prompt_btn.click(fn=lambda x: x + " specular", inputs=[relighting_prompt], outputs=[relighting_prompt])
                very_specular_prompt_btn.click(fn=lambda x: x + " very specular", inputs=[relighting_prompt], outputs=[relighting_prompt])
                with gr.Row():
                    clear_prompt_btn = gr.Button(value="Clear")
                    reuse_btn = gr.Button(value="Reuse Provisional Image Generation Prompt")
                clear_prompt_btn.click(fn=lambda x: "", inputs=[relighting_prompt], outputs=[relighting_prompt])
                reuse_btn.click(fn=lambda x: x, inputs=[prompt], outputs=[relighting_prompt])
            with gr.Accordion("Options", open=False):
                relighting_seed = gr.Number(value=3407, label="Seed", interactive=True)
                relighting_steps = gr.Number(value=20, label="Steps", interactive=True)
                relighting_cfg = gr.Number(value=3.0, label="CFG", interactive=True)
            relighting_generate_btn = gr.Button(value="Generate")

            def gen_relighting_image(masked_image, mask, res_folder_path, relighting_prompt, relighting_seed,
                                    relighting_steps, relighting_cfg, do_env_inpainting,
                                    progress=gr.Progress(track_tqdm=True)):
                relighting_gen(
                    masked_ref_img=masked_image,
                    mask=mask,
                    cond_path=res_folder_path,
                    frames=1,
                    prompt=relighting_prompt,
                    steps=int(relighting_steps),
                    seed=int(relighting_seed),
                    cfg=relighting_cfg
                )
                relit_img = imageio.v3.imread(res_folder_path + '/relighting00_0.png')
                if do_env_inpainting:
                    bg = imageio.v3.imread(res_folder_path + f'/bg00.png') / 255.
                else:
                    bg = np.zeros_like(relit_img)
                relit_img = relit_img / 255.
                mask_for_bg = imageio.v3.imread(res_folder_path + '/hint00_diffuse.png')[..., -1:] / 255.
                relit_img = relit_img * mask_for_bg + bg * (1. - mask_for_bg)
                relit_img = (relit_img * 255).clip(0, 255).astype(np.uint8)
                return relit_img

            relighting_generate_btn.click(fn=gen_relighting_image,
                                        inputs=[masked_image, mask, res_folder_path, relighting_prompt, relighting_seed,
                                                relighting_steps, relighting_cfg, is_env_lighting],
                                        outputs=[res_image])


if __name__ == '__main__':
    demo.queue().launch(server_name="0.0.0.0", share=True)
