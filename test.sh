# teaser
python3 infer_img.py --prov_img examples/provisional_img/futuristic_soldier.png --prompt "futuristic soldier with advanced armor weaponry and helmet" --env_map examples/env_map/grace.exr --out_vid ./output/soldier_grace.mp4
python3 infer_img.py --prov_img examples/provisional_img/futuristic_soldier.png --prompt "futuristic soldier with advanced armor weaponry and helmet" --env_map examples/env_map/kitchen.exr --out_vid ./output/soldier_kitchen.mp4
python3 infer_img.py --prov_img examples/provisional_img/rusty_frog.png --prompt "Rusty copper toy frog with spatially varying materials some parts are shinning other parts are rough" --env_map examples/env_map/grace.exr --out_vid ./output/frog_grace.mp4
python3 infer_img.py --prov_img examples/provisional_img/rusty_frog.png --prompt "Rusty copper toy frog with spatially varying materials some parts are shinning other parts are rough" --env_map examples/env_map/kitchen.exr --out_vid ./output/frog_kitchen.mp4

# main results
python3 infer_img.py --prov_img examples/provisional_img/machine-dragon-robot-in-platinum-0.png --mask_path examples/mask/machine-dragon-robot-in-platinum-0-mask.png --prompt "machine dragon robot in platinum" --env_map examples/env_map/grace.exr --out_vid ./output/machine_dragon_grace.mp4
python3 infer_img.py --prov_img examples/provisional_img/machine-dragon-robot-in-platinum-0.png --mask_path examples/mask/machine-dragon-robot-in-platinum-0-mask.png --prompt "machine dragon robot in platinum" --env_map examples/env_map/kitchen.exr --out_vid ./output/machine_dragon_kitchen.mp4
python3 infer_img.py --prov_img examples/provisional_img/machine-dragon-robot-in-platinum-0.png --mask_path examples/mask/machine-dragon-robot-in-platinum-0-mask.png --prompt "machine dragon robot in platinum" --env_map examples/env_map/rnl.exr --out_vid ./output/machine_dragon_rnl.mp4
python3 infer_img.py --prov_img examples/provisional_img/machine-dragon-robot-in-platinum-0.png --mask_path examples/mask/machine-dragon-robot-in-platinum-0-mask.png --prompt "machine dragon robot in platinum" --out_vid ./output/machine_dragon_pl.mp4

python3 infer_img.py --prov_img examples/provisional_img/gorgeous-ornate-fountain-made-of-marble.png --mask_path examples/mask/gorgeous-ornate-fountain-made-of-marble-mask.png --prompt "gorgeous ornate fountain made of marble" --env_map examples/env_map/grace.exr --out_vid ./output/fountain_grace.mp4
python3 infer_img.py --prov_img examples/provisional_img/gorgeous-ornate-fountain-made-of-marble.png --mask_path examples/mask/gorgeous-ornate-fountain-made-of-marble-mask.png --prompt "gorgeous ornate fountain made of marble" --env_map examples/env_map/kitchen.exr --out_vid ./output/fountain_kitchen.mp4
python3 infer_img.py --prov_img examples/provisional_img/gorgeous-ornate-fountain-made-of-marble.png --mask_path examples/mask/gorgeous-ornate-fountain-made-of-marble-mask.png --prompt "gorgeous ornate fountain made of marble" --env_map examples/env_map/rnl.exr --out_vid ./output/fountain_rnl.mp4
python3 infer_img.py --prov_img examples/provisional_img/gorgeous-ornate-fountain-made-of-marble.png --mask_path examples/mask/gorgeous-ornate-fountain-made-of-marble-mask.png --prompt "gorgeous ornate fountain made of marble" --out_vid ./output/fountain_pl.mp4

python3 infer_img.py --prov_img examples/provisional_img/stormtrooper-style-motorcycle.png --mask_path examples/mask/stormtrooper-style-motorcycle-mask.png --prompt "Storm trooper style motorcycle" --env_map examples/env_map/grace.exr --cfg 1.0 --out_vid ./output/motorcycle_grace.mp4
python3 infer_img.py --prov_img examples/provisional_img/stormtrooper-style-motorcycle.png --mask_path examples/mask/stormtrooper-style-motorcycle-mask.png --prompt "Storm trooper style motorcycle" --env_map examples/env_map/kitchen.exr --cfg 1.0 --out_vid ./output/motorcycle_kitchen.mp4
python3 infer_img.py --prov_img examples/provisional_img/stormtrooper-style-motorcycle.png --mask_path examples/mask/stormtrooper-style-motorcycle-mask.png --prompt "Storm trooper style motorcycle" --env_map examples/env_map/rnl.exr --cfg 1.0 --out_vid ./output/motorcycle_rnl.mp4
python3 infer_img.py --prov_img examples/provisional_img/stormtrooper-style-motorcycle.png --mask_path examples/mask/stormtrooper-style-motorcycle-mask.png --prompt "Storm trooper style motorcycle" --cfg 1.0 --out_vid ./output/motorcycle_pl.mp4

python3 infer_img.py --prov_img examples/provisional_img/girraffe_turtle.jpeg --prompt "A giraffe imitating a turtle, photorealistic" --env_map examples/env_map/grace.exr --out_vid ./output/giraffe_turtle_grace.mp4
python3 infer_img.py --prov_img examples/provisional_img/girraffe_turtle.jpeg --prompt "A giraffe imitating a turtle, photorealistic" --env_map examples/env_map/kitchen.exr --out_vid ./output/giraffe_turtle_kitchen.mp4
python3 infer_img.py --prov_img examples/provisional_img/girraffe_turtle.jpeg --prompt "A giraffe imitating a turtle, photorealistic" --env_map examples/env_map/rnl.exr --out_vid ./output/giraffe_turtle_rnl.mp4
python3 infer_img.py --prov_img examples/provisional_img/girraffe_turtle.jpeg --prompt "A giraffe imitating a turtle, photorealistic" --out_vid ./output/giraffe_turtle_pl.mp4

python3 infer_img.py --prov_img examples/provisional_img/rusty-phoenix.png --mask_path examples/mask/rusty-phoenix-mask.png --prompt "rusty sculpture of a phoenix with its head more polished yet the wings are more rusty" --env_map examples/env_map/grace.exr --out_vid ./output/phoenix_grace.mp4
python3 infer_img.py --prov_img examples/provisional_img/rusty-phoenix.png --mask_path examples/mask/rusty-phoenix-mask.png --prompt "rusty sculpture of a phoenix with its head more polished yet the wings are more rusty" --env_map examples/env_map/kitchen.exr --out_vid ./output/phoenix_kitchen.mp4
python3 infer_img.py --prov_img examples/provisional_img/rusty-phoenix.png --mask_path examples/mask/rusty-phoenix-mask.png --prompt "rusty sculpture of a phoenix with its head more polished yet the wings are more rusty" --env_map examples/env_map/rnl.exr --out_vid ./output/phoenix_rnl.mp4
python3 infer_img.py --prov_img examples/provisional_img/rusty-phoenix.png --mask_path examples/mask/rusty-phoenix-mask.png --prompt "rusty sculpture of a phoenix with its head more polished yet the wings are more rusty" --out_vid ./output/phoenix_pl.mp4

# single image relighting
python3 infer_img.py --prov_img examples/provisional_img/lion.png --prompt "a photo of a lion sculpture" --out_vid ./output/lion.mp4

# supp results
python3 infer_img.py --prov_img examples/provisional_img/starcraft-2-marine-machine-gun-0.png --prompt "starcraft 2 marine machine gun" --env_map examples/env_map/field.exr --out_vid ./output/starcraft_gun_field_0.mp4
python3 infer_img.py --prov_img examples/provisional_img/starcraft-2-marine-machine-gun-1.png --prompt "starcraft 2 marine machine gun" --env_map examples/env_map/field.exr --out_vid ./output/starcraft_gun_field_1.mp4
python3 infer_img.py --prov_img examples/provisional_img/starcraft-2-marine-machine-gun-2.png --prompt "starcraft 2 marine machine gun" --env_map examples/env_map/field.exr --out_vid ./output/starcraft_gun_field_2.mp4

python3 infer_img.py --prov_img examples/provisional_img/leather-glove-0.png --prompt "leather glove" --use_sam False --env_map examples/env_map/indoor-market.exr --out_vid ./output/leather_glove_market_0.mp4
python3 infer_img.py --prov_img examples/provisional_img/leather-glove-1.png --prompt "leather glove" --use_sam False --env_map examples/env_map/indoor-market.exr --out_vid ./output/leather_glove_market_1.mp4
python3 infer_img.py --prov_img examples/provisional_img/leather-glove-2.png --prompt "leather glove" --use_sam False --env_map examples/env_map/indoor-market.exr --out_vid ./output/leather_glove_market_2.mp4
python3 infer_img.py --prov_img examples/provisional_img/leather-glove-3.png --prompt "leather glove" --use_sam False --env_map examples/env_map/indoor-market.exr --out_vid ./output/leather_glove_market_3.mp4

python3 infer_img.py --prov_img examples/provisional_img/leather-glove-0.png --prompt "leather glove" --use_sam False --env_map examples/env_map/market2.exr --out_vid ./output/leather_glove_market2_0.mp4
python3 infer_img.py --prov_img examples/provisional_img/leather-glove-1.png --prompt "leather glove" --use_sam False --env_map examples/env_map/market2.exr --out_vid ./output/leather_glove_market2_1.mp4
python3 infer_img.py --prov_img examples/provisional_img/leather-glove-2.png --prompt "leather glove" --use_sam False --env_map examples/env_map/market2.exr --out_vid ./output/leather_glove_market2_2.mp4
python3 infer_img.py --prov_img examples/provisional_img/leather-glove-3.png --prompt "leather glove" --use_sam False --env_map examples/env_map/market2.exr --out_vid ./output/leather_glove_market2_3.mp4

python3 infer_img.py --prov_img examples/provisional_img/3d-animation-character-minimal-art-toy.png --prompt "3d animation character minimal art toy" --env_map examples/env_map/market2.exr --use_sam False --out_vid ./output/3d_animation_character_market2.mp4
python3 infer_img.py --prov_img examples/provisional_img/rusty-copper-toy-frog-with-spatially-varying-materials-some-parts-are-shinning-other-parts-are-rough.png --prompt "rusty copper toy frog with spatially varying materials some parts are shinning other parts are rough" --env_map examples/env_map/market2.exr --out_vid ./output/rusty_copper_toy_frog_market2.mp4

python3 infer_img.py --prov_img examples/provisional_img/stone-griffin.png --prompt "stone griffin" --env_map examples/env_map/park.exr --out_vid ./output/stone_griffin_park.mp4

python3 infer_img.py --prov_img examples/provisional_img/full-plate-armor-0.png --prompt "full plate armor" --use_sam False --env_map examples/env_map/cave.exr --out_vid ./output/full_plate_armor_cave_0.mp4
python3 infer_img.py --prov_img examples/provisional_img/full-plate-armor-1.png --prompt "full plate armor" --use_sam False --env_map examples/env_map/cave.exr --out_vid ./output/full_plate_armor_cave_1.mp4
python3 infer_img.py --prov_img examples/provisional_img/full-plate-armor-2.png --prompt "full plate armor" --use_sam False --env_map examples/env_map/cave.exr --out_vid ./output/full_plate_armor_cave_2.mp4

python3 infer_img.py --prov_img examples/provisional_img/caterpillar-work-boot.png --prompt "caterpillar work boot" --env_map examples/env_map/kitchen.exr --out_vid ./output/caterpillar_work_boot_kitchen.mp4

python3 infer_img.py --prov_img examples/provisional_img/a-decorated-plaster-rabbit-toy-plate-with-blue-fine-silk-ribbon-around-it.png --mask_path examples/mask/a-decorated-plaster-rabbit-toy-plate-with-blue-fine-silk-ribbon-around-it-mask.png --prompt "a decorated plaster rabbit toy plate with blue fine silk ribbon around it" --env_map examples/env_map/kitchen.exr --out_vid ./output/rabbit_plate_kitchen.mp4

python3 infer_img.py --prov_img examples/provisional_img/a-decorated-plaster-round-plate-with-blue-fine-silk-ribbon-around-it-0.png --mask_path examples/mask/a-decorated-plaster-round-plate-with-blue-fine-silk-ribbon-around-it-0-mask.png --prompt "a decorated plaster round plate with blue fine silk ribbon around it" --env_map examples/env_map/kitchen.exr --out_vid ./output/round_plate_kitchen_0.mp4
python3 infer_img.py --prov_img examples/provisional_img/a-decorated-plaster-round-plate-with-blue-fine-silk-ribbon-around-it-1.png --mask_path examples/mask/a-decorated-plaster-round-plate-with-blue-fine-silk-ribbon-around-it-1-mask.png --prompt "a decorated plaster round plate with blue fine silk ribbon around it" --env_map examples/env_map/galileo.exr --out_vid ./output/round_plate_galileo_1.mp4

python3 infer_img.py --prov_img examples/provisional_img/an-elephant-sculpted-from-plaster-and-the-elephant-nose-is-decorated-with-the-golden-texture.png --mask_path examples/mask/an-elephant-sculpted-from-plaster-and-the-elephant-nose-is-decorated-with-the-golden-texture-mask.png --prompt "an elephant sculpted from plaster and the elephant nose is decorated with the golden texture" --env_map examples/env_map/uffizi.exr --out_vid ./output/elephant_uffizi.mp4

python3 infer_img.py --prov_img examples/provisional_img/machine-dragon-robot-in-platinum-1.png --mask_path examples/mask/machine-dragon-robot-in-platinum-1-mask.png --prompt "machine dragon robot in platinum" --env_map examples/env_map/desert.exr --out_vid ./output/machine_dragon_desert_1.mp4
python3 infer_img.py --prov_img examples/provisional_img/machine-dragon-robot-in-platinum-2.png --mask_path examples/mask/machine-dragon-robot-in-platinum-2-mask.png --prompt "machine dragon robot in platinum" --env_map examples/env_map/desert.exr --out_vid ./output/machine_dragon_desert_2.mp4

python3 infer_img.py --prov_img examples/provisional_img/steampunk-space-tank-with-delicate-details-0.png --mask_path examples/mask/steampunk-space-tank-with-delicate-details-0-mask.png --prompt "steampunk space tank with delicate details" --env_map examples/env_map/city-night.exr --out_vid ./output/space_tank_city_night_0.mp4
python3 infer_img.py --prov_img examples/provisional_img/steampunk-space-tank-with-delicate-details-1.png --mask_path examples/mask/steampunk-space-tank-with-delicate-details-1-mask.png --prompt "steampunk space tank with delicate details" --env_map examples/env_map/city-night.exr --out_vid ./output/space_tank_city_night_1.mp4

# point light
python3 infer_img.py --prov_img examples/provisional_img/a-large-colorful-candle-high-quality-product-photo.png --prompt "a large colorful candle, high quality product photo" --out_vid ./output/candle_pl.mp4

# prompt material control
python3 infer_img.py --prov_img examples/provisional_img/futuristic_soldier.png --prompt "futuristic soldier with advanced armor weaponry and helmet, specular" --env_map examples/env_map/grace.exr --out_vid ./output/soldier_grace_specular.mp4
python3 infer_img.py --prov_img examples/provisional_img/futuristic_soldier.png --prompt "futuristic soldier with advanced armor weaponry and helmet, very specular" --env_map examples/env_map/grace.exr --out_vid ./output/soldier_grace_very_specular.mp4
python3 infer_img.py --prov_img examples/provisional_img/futuristic_soldier.png --prompt "futuristic soldier with advanced armor weaponry and helmet, metallic" --env_map examples/env_map/grace.exr --out_vid ./output/soldier_grace_metallic.mp4
python3 infer_img.py --prov_img examples/provisional_img/futuristic_soldier.png --prompt "futuristic soldier with advanced armor weaponry and helmet, metallic, very specular" --env_map examples/env_map/grace.exr --out_vid ./output/soldier_grace_metallic_very_specular.mp4


# examples/provisional_img/pottery.png
python3 infer_img.py --prov_img examples/provisional_img/pottery.png --prompt "a photo of a single pottery" --env_map examples/env_map/factory.exr --out_vid ./output/pottery_factory.mp4
python3 infer_img.py --prov_img examples/provisional_img/pottery.png --prompt "a photo of a single pottery, specular" --env_map examples/env_map/factory.exr --out_vid ./output/pottery_factory_specular.mp4
python3 infer_img.py --prov_img examples/provisional_img/pottery.png --prompt "a photo of a single pottery, very specular" --env_map examples/env_map/factory.exr --out_vid ./output/pottery_factory_very_specular.mp4
python3 infer_img.py --prov_img examples/provisional_img/pottery.png --prompt "a photo of a single pottery, metallic" --env_map examples/env_map/factory.exr --out_vid ./output/pottery_factory_metallic.mp4
python3 infer_img.py --prov_img examples/provisional_img/pottery.png --prompt "a photo of a single pottery, metallic, very specular" --env_map examples/env_map/factory.exr --out_vid ./output/pottery_factory_metallic_very_specular.mp4


# examples/provisional_img/wooden_car.png
python3 infer_img.py --prov_img examples/provisional_img/wooden_car.png --prompt "a photo of a wooden car" --env_map examples/env_map/chapel.exr --out_vid ./output/wooden_car_chapel.mp4
python3 infer_img.py --prov_img examples/provisional_img/wooden_car.png --prompt "a photo of a wooden car, specular" --env_map examples/env_map/chapel.exr --out_vid ./output/wooden_car_chapel_specular.mp4
python3 infer_img.py --prov_img examples/provisional_img/wooden_car.png --prompt "a photo of a wooden car, very specular" --env_map examples/env_map/chapel.exr --out_vid ./output/wooden_car_chapel_very_specular.mp4
python3 infer_img.py --prov_img examples/provisional_img/wooden_car.png --prompt "a photo of a wooden car, metallic" --env_map examples/env_map/chapel.exr --out_vid ./output/wooden_car_chapel_metallic.mp4
python3 infer_img.py --prov_img examples/provisional_img/wooden_car.png --prompt "a photo of a wooden car, metallic, very specular" --env_map examples/env_map/chapel.exr --out_vid ./output/wooden_car_chapel_metallic_very_specular.mp4


# gt mesh depth cond gen
# wolf head desk
python3 mesh_to_hints.py --mesh_path examples/depth_cond/mesh/wolf_head_desk.glb --output_dir tmp/wolf_head_desk
python3 infer_img.py --prov_img examples/depth_cond/prov_imgs/wolf_head_0.png --radiance_hints_path ./tmp/wolf_head_desk_fix/ --prompt "a wolf head sculpture and a vase on a desk" --out_vid output/wolf_head_desk_0.mp4
python3 infer_img.py --prov_img examples/depth_cond/prov_imgs/wolf_head_1.png --radiance_hints_path ./tmp/wolf_head_desk_fix/ --prompt "a wolf head sculpture and a vase on a desk" --out_vid output/wolf_head_desk_1.mp4
python3 infer_img.py --prov_img examples/depth_cond/prov_imgs/wolf_head_2.png --radiance_hints_path ./tmp/wolf_head_desk_fix/ --prompt "a wolf head sculpture and a vase on a desk, metallic, specular" --out_vid output/wolf_head_desk_2.mp4
python3 infer_img.py --prov_img examples/depth_cond/prov_imgs/wolf_head_3.png --radiance_hints_path ./tmp/wolf_head_desk_fix/ --prompt "a wolf head sculpture and a vase on a desk" --out_vid output/wolf_head_desk_3.mp4
python3 infer_img.py --prov_img examples/depth_cond/prov_imgs/wolf_head_4.png --radiance_hints_path ./tmp/wolf_head_desk_fix/ --prompt "a wolf head sculpture and a vase on a desk" --out_vid output/wolf_head_desk_4.mp4

# book shelf
python3 mesh_to_hints.py --mesh_path examples/mesh/book_shelf.glb --output_dir tmp/book_shelf --env_map examples/env_map/studio.exr --cam_azi 180 --cam_elev 20 --cam_dist 0.5 --cam_fov 60 --spp 1024
python3 infer_img.py --prov_img examples/depth_cond/prov_imgs/book_shelf_0.png --radiance_hints_path ./tmp/book_shelf/ --prompt "a wooden shelf" --out_vid output/book_shelf_0.mp4
python3 infer_img.py --prov_img examples/depth_cond/prov_imgs/book_shelf_1.png --radiance_hints_path ./tmp/book_shelf/ --prompt "a rusty book shelf" --out_vid output/book_shelf_1.mp4
python3 infer_img.py --prov_img examples/depth_cond/prov_imgs/book_shelf_2.png --radiance_hints_path ./tmp/book_shelf/ --prompt "a clay shelf" --out_vid output/book_shelf_2.mp4
python3 infer_img.py --prov_img examples/depth_cond/prov_imgs/book_shelf_3.png --radiance_hints_path ./tmp/book_shelf/ --prompt "a metallic shelf, specular" --out_vid output/book_shelf_3.mp4
python3 infer_img.py --prov_img examples/depth_cond/prov_imgs/book_shelf_4.png --radiance_hints_path ./tmp/book_shelf/ --prompt "a colorful shelf, very specular" --out_vid output/book_shelf_4.mp4

# chair
python3 mesh_to_hints.py --mesh_path examples/mesh/chair.glb --output_dir tmp/chair --cam_azi 150 --env_map examples/env_map/garage.exr --cam_elev 20
python3 infer_img.py --prov_img examples/depth_cond/prov_imgs/chair_0.png --radiance_hints_path ./tmp/chair/ --prompt "a wooden chair" --out_vid output/chair_0.mp4
python3 infer_img.py --prov_img examples/depth_cond/prov_imgs/chair_1.png --radiance_hints_path ./tmp/chair/ --prompt "a metallic chair, specular" --out_vid output/chair_1.mp4
python3 infer_img.py --prov_img examples/depth_cond/prov_imgs/chair_2.png --radiance_hints_path ./tmp/chair/ --prompt "a rusty chair" --out_vid output/chair_2.mp4
python3 infer_img.py --prov_img examples/depth_cond/prov_imgs/chair_3.png --radiance_hints_path ./tmp/chair/ --prompt "a clay chair" --out_vid output/chair_3.mp4
python3 infer_img.py --prov_img examples/depth_cond/prov_imgs/chair_4.png --radiance_hints_path ./tmp/chair/ --prompt "a clay chair" --out_vid output/chair_4.mp4
