import json
import math
import os
from dataclasses import dataclass, field
import random
from typing import Optional, List

import imageio
import numpy as np
import simple_parsing
from tqdm import tqdm


@dataclass
class Options:
    """ 3D dataset rendering script """
    mesh_path: str  # Base path to 3D models
    output_dir: str = 'tmp/mesh_render'  # Output directory
    cam_elev: float = 30.  # Elevation of the camera in degrees
    cam_azi: float = 0.    # Azimuth of the camera in degrees
    cam_dist: float = 1.5  # Distance of the camera from the origin
    cam_fov: float = 35.  # Field of view of the camera
    pl_pos_r: float = 2.  # Rotation radius of the point light
    pl_pos_center: List[float] = field(default_factory=lambda: [0., 0., 1.])  # Center position of the point light
    power: float = 100.  # Power of the point light
    frames: int = 120     # Number of frames for lighting controlled video
    env_map: Optional[str] = None  # Path to env map
    spp: int = 512        # Samples per pixel


def render_core(args: Options):
    import bpy

    from bpy_helper.camera import create_camera, look_at_to_c2w
    from bpy_helper.io import render_depth_map, mat2list, array2list, save_blend_file
    from bpy_helper.light import create_point_light, set_env_light, create_area_light
    from bpy_helper.material import create_white_diffuse_material, create_specular_ggx_material, clear_emission_and_alpha_nodes, create_invisible_material
    from bpy_helper.scene import import_3d_model, normalize_scene, reset_scene
    from bpy_helper.utils import stdout_redirected

    def render_rgb_and_hint(output_path):
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

    def configure_blender():
        # Set the render resolution
        bpy.context.scene.render.resolution_x = 512
        bpy.context.scene.render.resolution_y = 512
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = args.spp
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

        # Enable the alpha channel for GT mask
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    reset_scene()

    # Import the 3D object
    with stdout_redirected():
        import_3d_model(args.mesh_path)
    normalize_scene(use_bounding_sphere=True)
    clear_emission_and_alpha_nodes()

    # Configure blender
    configure_blender()

    # 0. Place Camera, render gt depth map
    radius = args.cam_dist
    elev = args.cam_elev
    azi = args.cam_azi
    x = radius * math.cos(math.radians(elev)) * math.sin(math.radians(azi))
    y = radius * math.cos(math.radians(elev)) * math.cos(math.radians(azi))
    z = radius * math.sin(math.radians(elev))
    eye = np.array([x, y, z])
    c2w = look_at_to_c2w(eye)
    camera = create_camera(c2w, args.cam_fov)
    bpy.context.scene.camera = camera
    with stdout_redirected():
        render_depth_map(args.output_dir)
        bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
        bpy.context.scene.world.node_tree.nodes["Background"].inputs[1].default_value = 2.0
        bpy.context.scene.render.filepath = f'{args.output_dir}/default_pov.png'
        bpy.ops.render.render(animation=False, write_still=True)

    # Render radiance hints
    frames = args.frames
    if args.env_map is not None:
        invisible_mat = create_invisible_material()
        for i in tqdm(range(args.frames), desc='Rendering Env Map Hints'):
            z_angle = (i / args.frames) * np.pi * 2.
            set_env_light(args.env_map, rotation_euler=[0, 0, z_angle])

            bpy.context.scene.render.film_transparent = True
            bpy.context.scene.render.image_settings.color_mode = 'RGBA'
            with stdout_redirected():
                render_rgb_and_hint(args.output_dir + f'/hint{i:02d}')

            bpy.context.scene.view_layers["ViewLayer"].material_override = invisible_mat
            bpy.context.scene.render.film_transparent = False
            bpy.context.scene.render.image_settings.color_mode = 'RGB'
            with stdout_redirected():
                bpy.context.scene.render.filepath = args.output_dir + f'/bg{i:02d}.png'
                bpy.ops.render.render(animation=False, write_still=True)
    else:
        pls = [(
            args.pl_pos_r * np.sin(frame / frames * np.pi * 2.) + args.pl_pos_center[0],
            args.pl_pos_r * np.cos(frame / frames * np.pi * 2.) + args.pl_pos_center[1],
            args.pl_pos_center[2]
        ) for frame in range(frames)]
        for pl_idx, pl in tqdm(list(enumerate(pls))):
            pl_pos = pl
            _point_light = create_point_light(pl_pos, args.power)
            # save_blend_file('debug.blend')
            # exit(0)
            with stdout_redirected():
                render_rgb_and_hint(args.output_dir + f'/hint{pl_idx:02d}')


if __name__ == '__main__':
    args: Options = simple_parsing.parse(Options)
    print("options:", args)
    render_core(args)
