import os
import multiprocessing
import tempfile
from multiprocessing import Process
from typing import Optional

from tqdm import tqdm


def render_hint_images(model_path, fov, pls, power=500., geo_smooth=True, output_folder: Optional[str] = None,
                       env_map: Optional[str] = None, env_start_azi=0., resolution=512, use_gpu=False):
    import bpy
    import numpy as np

    from bpy_helper.camera import create_camera
    from bpy_helper.light import set_env_light, create_point_light
    from bpy_helper.material import create_white_diffuse_material, create_specular_ggx_material
    from bpy_helper.scene import reset_scene, import_3d_model
    from bpy_helper.utils import stdout_redirected

    def configure_blender():
        # Set the render resolution
        bpy.context.scene.render.resolution_x = resolution
        bpy.context.scene.render.resolution_y = resolution
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 512
        if use_gpu:
            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            bpy.context.scene.cycles.device = 'GPU'
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
            bpy.context.scene.render.threads = 8
            bpy.context.scene.render.threads_mode = 'FIXED'

        # Enable the alpha channel for GT mask
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'

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

            # and png
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.context.scene.render.filepath = os.path.abspath(f'{output_path}{mat_name}.png')
            bpy.ops.render.render(animation=False, write_still=True)

    # Render hints
    reset_scene()
    import_3d_model(model_path)
    if geo_smooth:
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.modifiers.new("Smooth", type="SMOOTH")
                smooth_modifier = obj.modifiers["Smooth"]
                smooth_modifier.factor = 0.5
                smooth_modifier.iterations = 50
    configure_blender()

    c2w = np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ])
    camera = create_camera(c2w, fov)
    bpy.context.scene.camera = camera
    if output_folder is None:
        output_folder = tempfile.mkdtemp()
    for i in tqdm(range(len(pls)), desc='Rendering Hints'):
        if env_map:
            z_angle = (i / len(pls) + env_start_azi) * np.pi * 2.
            set_env_light(env_map, rotation_euler=[0, 0, z_angle])
        else:
            pl_pos = pls[i]
            _point_light = create_point_light(pl_pos, power)

        with stdout_redirected():
            render_rgb_and_hint(output_folder + f'/hint{i:02d}')

    return output_folder


def render_bg_images(fov, pls, output_folder: Optional[str] = None, env_map: Optional[str] = None, env_start_azi=0., resolution=512, use_gpu=False):
    import bpy
    import numpy as np

    from bpy_helper.camera import create_camera
    from bpy_helper.light import set_env_light
    from bpy_helper.scene import reset_scene
    from bpy_helper.utils import stdout_redirected

    def configure_blender():
        # Set the render resolution
        bpy.context.scene.render.resolution_x = resolution
        bpy.context.scene.render.resolution_y = resolution
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 512

        # Enable the alpha channel for GT mask
        bpy.context.scene.render.film_transparent = False
        bpy.context.scene.render.image_settings.color_mode = 'RGB'

        if use_gpu:
            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            bpy.context.scene.cycles.device = 'GPU'
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
            bpy.context.scene.render.threads = 8
            bpy.context.scene.render.threads_mode = 'FIXED'

    def render_env_bg(output_path):
        bpy.context.scene.view_layers["ViewLayer"].material_override = None
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = f'{output_path}.png'
        bpy.ops.render.render(animation=False, write_still=True)

    # Render backgrounds
    reset_scene()
    configure_blender()

    c2w = np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ])
    camera = create_camera(c2w, fov)
    bpy.context.scene.camera = camera
    if output_folder is None:
        output_folder = tempfile.mkdtemp()
    for i in tqdm(range(len(pls)), desc='Rendering Env Backgrounds'):
        z_angle = (i / len(pls) + env_start_azi) * np.pi * 2.
        set_env_light(env_map, rotation_euler=[0, 0, z_angle])

        with stdout_redirected():
            render_env_bg(output_folder + f'/bg{i:02d}')

    return output_folder


def render_hint_images_wrapper(model_path, fov, pls, power, geo_smooth, output_folder, env_map, env_start_azi, resolution, return_dict):
    output_folder = render_hint_images(model_path, fov, pls, power, geo_smooth, output_folder, env_map, env_start_azi, resolution)
    if env_map is not None:
        bg_output_folder = render_bg_images(fov, pls, output_folder, env_map, env_start_azi, resolution)
        return_dict['bg_output_folder'] = bg_output_folder
    return_dict['output_folder'] = output_folder


def render_hint_images_btn_func(model_path, fov, pls, power=500., geo_smooth=True, output_folder: Optional[str] = None,
                                env_map: Optional[str] = None, env_start_azi=0., resolution=512):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = Process(target=render_hint_images_wrapper, args=(model_path, fov, pls, power, geo_smooth, output_folder, env_map, env_start_azi, resolution, return_dict))
    p.start()
    p.join()
    return return_dict['output_folder']
