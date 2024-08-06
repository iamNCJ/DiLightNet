import json
import math
import os
from dataclasses import dataclass
import random
from typing import Optional

import imageio
import numpy as np
import simple_parsing


@dataclass
class Options:
    """ 3D dataset rendering script """
    three_d_model_path: str  # Base path to 3D models
    env_map_list_json: str = './assets/hdri/polyhaven_hdris.json'  # Path to env map list
    env_map_dir_path: str = './assets/hdri/files'  # Path to env map directory
    white_env_map_dir_path: str = './assets/hdri/file_bw'  # Path to white env map directory
    output_dir: str = './output'  # Output directory
    num_views: int = 2  # Number of views
    num_white_pls: int = 3  # Number of white point lighting
    num_rgb_pls: int = 0  # Number of RGB point lighting
    num_multi_pls: int = 3  # Number of multi point lighting
    max_pl_num: int = 3  # Maximum number of point lights
    num_white_envs: int = 3  # Number of white env lighting
    num_env_lights: int = 3  # Number of env lighting
    num_area_lights: int = 3  # Number of area lights
    seed: Optional[int] = None  # Random seed


def render_core(args: Options):
    import bpy

    from bpy_helper.camera import create_camera, look_at_to_c2w
    from bpy_helper.io import render_depth_map, mat2list, array2list
    from bpy_helper.light import create_point_light, set_env_light, create_area_light
    from bpy_helper.material import create_white_diffuse_material, create_specular_ggx_material, clear_emission_and_alpha_nodes
    from bpy_helper.random import gen_random_pts_around_origin
    from bpy_helper.scene import import_3d_model, normalize_scene, reset_scene
    from bpy_helper.utils import stdout_redirected

    def render_rgb_and_hint(output_path):
        bpy.context.scene.view_layers["ViewLayer"].material_override = None
        bpy.context.scene.render.image_settings.file_format = 'PNG'  # set output to png (with tonemapping)
        bpy.context.scene.render.filepath = f'{output_path}.png'
        bpy.ops.render.render(animation=False, write_still=True)
        img = imageio.v3.imread(f'{output_path}.png') / 255.
        if img.shape[-1] == 4:
            img = img[..., :3] * img[..., 3:]  # fix edge aliasing
        imageio.v3.imwrite(f'{output_path}.png', (img * 255).clip(0, 255).astype(np.uint8))

        MAT_DICT = {
            '_diffuse': create_white_diffuse_material(),
            '_ggx0.05': create_specular_ggx_material(0.05),
            '_ggx0.13': create_specular_ggx_material(0.13),
            '_ggx0.34': create_specular_ggx_material(0.34),
        }

        # render
        for mat_name, mat in MAT_DICT.items():
            bpy.context.scene.view_layers["ViewLayer"].material_override = mat
            bpy.context.scene.render.filepath = f'{output_path}{mat_name}.png'
            bpy.ops.render.render(animation=False, write_still=True)
            img = imageio.v3.imread(f'{output_path}{mat_name}.png') / 255.
            if img.shape[-1] == 4:
                img = img[..., :3] * img[..., 3:]  # fix edge aliasing
            imageio.v3.imwrite(f'{output_path}{mat_name}.png', (img * 255).clip(0, 255).astype(np.uint8))

    def configure_blender():
        # Set the render resolution
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

        # Enable the alpha channel for GT mask
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    reset_scene()

    # Import the 3D object
    file_path = args.three_d_model_path
    with stdout_redirected():
        import_3d_model(file_path)
    scale, offset = normalize_scene(use_bounding_sphere=True)
    clear_emission_and_alpha_nodes()

    # Configure blender
    configure_blender()

    # Load env map list
    env_map_list = json.load(open(args.env_map_list_json, 'r'))

    # Render GT images & hints
    seed_view = None if args.seed is None else args.seed
    seed_white_pl = None if args.seed is None else args.seed + 1
    seed_rgb_pl = None if args.seed is None else args.seed + 2
    seed_multi_pl = None if args.seed is None else args.seed + 3
    seed_area = None if args.seed is None else args.seed + 4
    res_dir = f"{args.output_dir}/{file_path.split('/')[-1].split('.')[0]}"
    os.makedirs(res_dir, exist_ok=True)
    json.dump({'scale': scale, 'offset': array2list(offset)}, open(f'{res_dir}/normalize.json', 'w'), indent=4)

    eyes = gen_random_pts_around_origin(
        seed=seed_view,
        N=args.num_views,
        min_dist_to_origin=1.0,
        max_dist_to_origin=1.0,
        min_theta_in_degree=10,
        max_theta_in_degree=90
    )
    for eye_idx in range(args.num_views):
        # 0. Place Camera, render gt depth map
        eye = eyes[eye_idx]
        fov = random.uniform(25, 35)
        radius = random.uniform(0.8, 1.1) * (0.5 / math.tanh(fov / 2. * (math.pi / 180.)))
        eye = [x * radius for x in eye]
        c2w = look_at_to_c2w(eye)
        camera = create_camera(c2w, fov)
        bpy.context.scene.camera = camera
        view_path = f'{res_dir}/view_{eye_idx}'
        os.makedirs(view_path, exist_ok=True)
        with stdout_redirected():
            render_depth_map(view_path)
        # save cam info
        json.dump({'c2w': mat2list(c2w), 'fov': fov}, open(f'{view_path}/cam.json', 'w'), indent=4)

        # 1. Single white point light
        white_pls = gen_random_pts_around_origin(
            seed=seed_white_pl,
            N=args.num_white_pls,
            min_dist_to_origin=4.0,
            max_dist_to_origin=5.0,
            min_theta_in_degree=0,
            max_theta_in_degree=60
        )
        for white_pl_idx in range(args.num_white_pls):
            pl = white_pls[white_pl_idx]
            power = random.uniform(500, 1500)
            _point_light = create_point_light(pl, power)
            ref_pl_path = f'{view_path}/white_pl_{white_pl_idx}'
            os.makedirs(ref_pl_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{ref_pl_path}/gt')
            # save point light info
            json.dump({
                'pos': array2list(pl),
                'power': power,
            }, open(f'{ref_pl_path}/white_pl.json', 'w'), indent=4)

        # 2. Single RGB point light
        rgb_pls = gen_random_pts_around_origin(
            seed=seed_rgb_pl,
            N=args.num_rgb_pls,
            min_dist_to_origin=4.0,
            max_dist_to_origin=5.0,
            min_theta_in_degree=0,
            max_theta_in_degree=60
        )
        for rgb_pl_idx in range(args.num_rgb_pls):
            pl = rgb_pls[rgb_pl_idx]
            power = random.uniform(900, 1500)  # slightly brighter than white light
            rgb = [random.uniform(0, 1) for _ in range(3)]
            _point_light = create_point_light(pl, power, rgb=rgb)
            ref_pl_path = f'{view_path}/rgb_pl_{rgb_pl_idx}'
            os.makedirs(ref_pl_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{ref_pl_path}/gt')
            # save point light info
            json.dump({
                'pos': array2list(pl),
                'power': power,
                'color': rgb,
            }, open(f'{ref_pl_path}/rgb_pl.json', 'w'), indent=4)

        # 3. Multi point lights
        multi_pls = gen_random_pts_around_origin(
            seed=seed_multi_pl,
            N=args.num_multi_pls * args.max_pl_num,
            min_dist_to_origin=4.0,
            max_dist_to_origin=5.0,
            min_theta_in_degree=0,
            max_theta_in_degree=60
        )
        for multi_pl_idx in range(args.num_multi_pls):
            pls = multi_pls[multi_pl_idx * args.max_pl_num: (multi_pl_idx + 1) * args.max_pl_num]
            powers = [random.uniform(500, 1500) for _ in range(args.max_pl_num)]
            for pl_idx in range(args.max_pl_num):
                _point_light = create_point_light(pls[pl_idx], powers[pl_idx], keep_other_lights=pl_idx > 0)
            ref_pl_path = f'{view_path}/multi_pl_{multi_pl_idx}'
            os.makedirs(ref_pl_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{ref_pl_path}/gt')
            # save point light info
            json.dump({
                'pos': mat2list(pls),
                'power': powers,
            }, open(f'{ref_pl_path}/multi_pl.json', 'w'), indent=4)

        # 4. White env lighting
        for env_idx in range(args.num_white_envs):
            env_map = random.choice(env_map_list)
            env_map_path = f'{args.white_env_map_dir_path}/{env_map}.exr'
            rotation_euler = [0, 0, random.uniform(-math.pi, math.pi)]
            strength = 1.0
            set_env_light(env_map_path, rotation_euler=rotation_euler, strength=strength)
            env_path = f'{view_path}/white_env_{env_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}/gt')
            # save env map info
            json.dump({
                'env_map': env_map,
                'rotation_euler': rotation_euler,
                'strength': strength,
            }, open(f'{env_path}/white_env.json', 'w'), indent=4)

        # 5. Env lighting
        for env_map_idx in range(args.num_env_lights):
            env_map = random.choice(env_map_list)
            env_map_path = f'{args.env_map_dir_path}/{env_map}.exr'
            rotation_euler = [0, 0, random.uniform(-math.pi, math.pi)]
            strength = 1.0  # random.uniform(0.8, 1.2)
            set_env_light(env_map_path, rotation_euler=rotation_euler, strength=strength)
            env_path = f'{view_path}/env_{env_map_idx}'
            os.makedirs(env_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{env_path}/gt')
            # save env map info
            json.dump({
                'env_map': env_map,
                'rotation_euler': rotation_euler,
                'strength': strength,
            }, open(f'{env_path}/env.json', 'w'), indent=4)
        
        # 6. Area lighting
        area_light_positions = gen_random_pts_around_origin(
            seed=seed_area,
            N=args.num_area_lights,
            min_dist_to_origin=4.0,
            max_dist_to_origin=5.0,
            min_theta_in_degree=0,
            max_theta_in_degree=60
        )
        for area_light_idx in range(args.num_area_lights):
            area_light_pos = area_light_positions[area_light_idx]
            area_light_power = random.uniform(700, 1500)
            area_light_size = random.uniform(5., 10.)
            _area_light = create_area_light(area_light_pos, area_light_power, area_light_size)
            area_path = f'{view_path}/area_{area_light_idx}'
            os.makedirs(area_path, exist_ok=True)
            with stdout_redirected():
                render_rgb_and_hint(f'{area_path}/gt')
            # save area light info
            json.dump({
                'pos': array2list(area_light_pos),
                'power': area_light_power,
                'size': area_light_size,
            }, open(f'{area_path}/area.json', 'w'), indent=4)


if __name__ == '__main__':
    args: Options = simple_parsing.parse(Options)
    print("options:", args)
    render_core(args)
