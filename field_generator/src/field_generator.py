import torch
from torchvision.transforms import v2
import numpy as np
import time
import random
from src.utils import gaussian_filter, normalize_tensor, normalize_positive_tensor
from types import SimpleNamespace
from src.generators import generate_probability_maps,\
                       generate_tree_offsets,\
                       generate_tree_sprites,\
                       prob_to_treesize,\
                       generate_field_mask,\
                       project_shadow,\
                       compute_gradient_map,\
                       apply_lighting,\
                       bool_tensor_to_coords, \
                       generate_rectangle_coords, \
                       coords_map_to_list, \
                       soft_rectangle_mask

def generate_field(**kwargs):
    config = SimpleNamespace(**kwargs)

    ############################### Background color
    if config.verbose:
        tic = time.perf_counter()

    GAIN = config.field_generation_zoom
    CANVAS_SIZE = config.field_size[0]*GAIN, config.field_size[1]*GAIN

    def get_random_value(value_list):
        return_value = value_list
        if type(value_list) is list:
            return_value = random.choice(value_list)
        return return_value

    # Pick color set
    color_set = random.choice(config.color_sets)
    color_bkg = [c*(1+random.uniform(-config.color_randomization_strength, config.color_randomization_strength)) for c in color_set[0]]
    color_bkg_overlay = [c*(1+random.uniform(-config.color_randomization_strength, config.color_randomization_strength)) for c in color_set[1]]
    color_bkg_patches = [c*(1+random.uniform(-config.color_randomization_strength, config.color_randomization_strength)) for c in color_set[2]]
    color_bkg_patches_notrees = [c*(1+random.uniform(-config.color_randomization_strength, config.color_randomization_strength)) for c in color_set[3]]
    color_tree = [c*(1+random.uniform(-config.color_randomization_strength, config.color_randomization_strength)) for c in color_set[4]]
    color_tree_shadow = [c*(1+random.uniform(-config.color_randomization_strength, config.color_randomization_strength)) for c in color_set[5]]

    # generate background color
    canvas_shape = (config.batch_size,
                    CANVAS_SIZE[0],
                    CANVAS_SIZE[1])

    field_channels = torch.stack([
        torch.full(canvas_shape, color_bkg[0], device=config.device, dtype=torch.float32),
        torch.full(canvas_shape, color_bkg[1], device=config.device, dtype=torch.float32),
        torch.full(canvas_shape, color_bkg[2], device=config.device, dtype=torch.float32),
        torch.full(canvas_shape, color_bkg[3], device=config.device, dtype=torch.float32)
    ])

    if config.verbose:
        toc = time.perf_counter()
        print(f"Background color generation time: {toc - tic:0.4f} seconds")

    ############################### Apply patches and overlay to background

    if config.verbose:
        tic = time.perf_counter()

    # Apply multiple passes of patches to the background
    passes_number = random.randint(config.bkg_patches_passes_min, config.bkg_patches_passes_max)

    for _ in range(passes_number):
        # Generate coordinates
        minmax_sizes = random.choice(config.bkg_patches_minmax_sizes)
        patches_coords = generate_rectangle_coords(batch_size=config.batch_size,
                                                   canvas_size=CANVAS_SIZE,
                                                   minmax_width=minmax_sizes[0:2]*GAIN,
                                                   minmax_height=minmax_sizes[2:4]*GAIN,
                                                   device=config.device,)
        # Generate patch
        patch_mask = soft_rectangle_mask(CANVAS_SIZE[0], CANVAS_SIZE[1], patches_coords, sharpness=config.bkg_patches_steepness, device=config.device)
        # Apply patch to background
        for i in range(4):
            field_channels[i] = field_channels[i]*(1-patch_mask) + color_bkg_patches[i]*patch_mask

        # Generate overlay mask
        picked_color = color_bkg_overlay
        filter_size = get_random_value(config.bkg_overlay_filter_size)
        filter_sigma = get_random_value(config.bkg_overlay_filter_sigma)
        overlay_mask = torch.tanh(torch.randn_like(field_channels[0]))
        overlay_mask = torch.sigmoid(normalize_tensor(gaussian_filter(overlay_mask, filter_size*GAIN, filter_sigma*GAIN))*config.bkg_overlay_steepness)*random.uniform(0.0, 1.0)*patch_mask
        # Apply overlay to background
        for i in range(4):
            field_channels[i] = field_channels[i]*(1-overlay_mask) + color_bkg_overlay[i]*overlay_mask

        del patch_mask, overlay_mask, patches_coords

    if config.verbose:
        toc = time.perf_counter()
        print(f"Background patches and overlay generation time: {toc - tic:0.4f} seconds")

    ############################### Generate field of trees
    if config.verbose:
        tic = time.perf_counter()

    # Generate probability maps
    prob_map = generate_probability_maps(prob_map_size=config.treemap_size,
                                         batch_size=config.batch_size,
                                         filter_size=config.treemap_filter_size,
                                         filter_sigma=config.treemap_filter_sigma,
                                         noise_level=config.treemap_noise_strength,
                                         sparse_remove_rate=config.treemap_sparse_remove_rate,
                                         sparse_add_rate=config.treemap_sparse_add_rate,
                                         row_prune_rate=config.treemap_row_prune_rate,
                                         column_prune_rate=config.treemap_column_prune_rate,
                                         final_filter_size=config.treemap_final_filter_size,
                                         final_filter_sigma=config.treemap_final_filter_sigma,
                                         device=config.device)

    # Generate tree size map and number of trees labels
    treesize_map, tree_boolmap = prob_to_treesize(prob_map,
                                                  threshold=config.tree_threshold,
                                                  treeshape_min_size=config.treeshape_min_radius*GAIN,
                                                  treeshape_max_size=config.treeshape_max_radius*GAIN,
                                                  steepness=config.tree_steepness,
                                                  distribution_shift=config.tree_distribution_shift)

    # Generate tree offsets and absolute positions
    tree_offsets = generate_tree_offsets(treesize_map=treesize_map,
                                         center_jitter=(config.tree_center_jitter*GAIN, config.tree_center_jitter*GAIN),
                                         alternate_offset=config.tree_alternate_offset*GAIN)


    # Free gpu memory
    del prob_map

    if config.verbose:
        toc = time.perf_counter()
        print(f"Field of trees probability map generation time: {toc - tic:0.4f} seconds")

    ############################### Generate coordinates and sizes
    if config.verbose:
        tic = time.perf_counter()

    # Convert treebool_map to coordinate labels
    constant_offset = [config.tree_offset*GAIN + config.tree_sprite_size*GAIN]*2
    tree_coords_map = bool_tensor_to_coords(tree_boolmap,
                                            tree_offsets,
                                            constant_offset,
                                            config.tree_xspace*GAIN,
                                            config.tree_yspace*GAIN,)

    if config.verbose:
        toc = time.perf_counter()
        print(f"Field of trees coordinates generation time: {toc - tic:0.4f} seconds")

    ############################### Generate patches without overlapping trees

    if config.verbose:
        tic = time.perf_counter()

    # Apply multiple passes of patches to the background
    passes_number = random.randint(config.bkg_patches_notrees_passes_min, config.bkg_patches_notrees_passes_max)

    picked_color = color_bkg_patches_notrees
    for _ in range(passes_number):
        # Generate coordinates
        minmax_size = random.choice(config.bkg_patches_notrees_minmax_sizes)
        patches_coords = generate_rectangle_coords(batch_size=config.batch_size,
                                                   canvas_size=CANVAS_SIZE,
                                                   minmax_width=minmax_size[0:2]*GAIN,
                                                   minmax_height=minmax_size[2:4]*GAIN,
                                                   device=config.device,)
        # Generate patch
        patch_mask = soft_rectangle_mask(CANVAS_SIZE[0], CANVAS_SIZE[1], patches_coords, sharpness=config.bkg_patches_steepness, device=config.device)
        # Apply patch to background
        for i in range(4):
            field_channels[i] = field_channels[i]*(1-patch_mask) + picked_color[i]*patch_mask

        # Remove trees overlapping with the patch
        for i in range(len(tree_coords_map)):
            # Remove the trees that overlap with the patch
            selector = (tree_coords_map[i][0] >= patches_coords[i][0]) & \
                       (tree_coords_map[i][1] >= patches_coords[i][1]) & \
                       (tree_coords_map[i][0] < patches_coords[i][2]) & \
                       (tree_coords_map[i][1] < patches_coords[i][3])
            treesize_map[i][selector] = 0.0
            tree_boolmap[i][selector] = False

        del patch_mask, patches_coords

    if config.verbose:
        toc = time.perf_counter()
        print(f"Background patches and overlay generation time (without overlapping trees): {toc - tic:0.4f} seconds")

    ############################### Generate tree coordinates list
    if config.verbose:
        tic = time.perf_counter()
    # Generate tree coordinates list
    tree_coordinates = coords_map_to_list(tree_coords_map,
                                          tree_boolmap,
                                          treesize_map*2)

    ############################### Apply noise to background
    if config.verbose:
        tic = time.perf_counter()

    # apply noise to bkg
    noise = torch.randn_like(field_channels[0])*config.bkg_noise_strength
    field_channels *= (1+noise)

    # free gpu memory
    del noise

    if config.verbose:
        toc = time.perf_counter()
        print(f"Background noise generation time: {toc - tic:0.4f} seconds")

    ############################### Generate tree sprites

    if config.verbose:
        tic = time.perf_counter()

    # Generate tree sprites
    tree_sprites = generate_tree_sprites(treesize_map=treesize_map,
                                         max_tree_size=config.treeshape_max_radius*GAIN,
                                         tree_sprite_size=config.tree_sprite_size*GAIN,
                                         tree_offsets=tree_offsets,
                                         noise_level=config.treeshape_noise,
                                         filter_size=config.treeshape_filter_size*GAIN,
                                         filter_sigma=config.treeshape_filter_sigma*GAIN,)

    # Free gpu memory
    del treesize_map

    # Generate field tree mask
    field_tree_mask = generate_field_mask(tree_sprites,
                                          CANVAS_SIZE,
                                          distance=(config.tree_xspace*GAIN, config.tree_yspace*GAIN),
                                          offset=config.tree_offset*GAIN)

    # Free gpu memory
    del tree_sprites

    if config.verbose:
        toc = time.perf_counter()
        print(f"Field of trees generation time: {toc - tic:0.4f} seconds")

    ############################### Generate and apply projected shadows
    if config.shadow_length > 0:
        if config.verbose:
            tic = time.perf_counter()

        # Generate shadows mask
        field_shadows_mask = project_shadow(field_tree_mask,
                                            config.shadow_iterations,
                                            config.shadow_length*GAIN,
                                            config.shadow_direction)

        # apply projected shadows to backgroud
        for i in range(4):
            field_channels[i] = field_channels[i]*(1-field_shadows_mask) + color_tree_shadow[i]*field_shadows_mask

        if config.verbose:
            toc = time.perf_counter()
            print(f"Projected shadows generation time: {toc - tic:0.4f} seconds")

    ############################### Background gaussian blur
    if config.verbose:
        tic = time.perf_counter()

    # smooth background
    field_channels = gaussian_filter(field_channels, 5, 1.5)

    if config.verbose:
        toc = time.perf_counter()
        print(f"Background gaussian blur time: {toc - tic:0.4f} seconds")

    ############################### Put trees on the field
    if config.verbose:
        tic = time.perf_counter()

    # Put trees on the field
    for i in range(4):
        field_channels[i] = field_channels[i]*(1-field_tree_mask) + color_tree[i]*field_tree_mask

    if config.verbose:
        toc = time.perf_counter()
        print(f"Trees generation time: {toc - tic:0.4f} seconds")

    ############################### Apply shadowing on the trees
    if config.verbose:
        tic = time.perf_counter()

    # Compute field gradient
    field_gradient = compute_gradient_map(field_tree_mask,
                                          config.shadow_direction)

    # free gpu memory
    del field_tree_mask

    # Apply shadowing based on gradient map
    apply_lighting(field_channels, field_gradient, config.shadow_blending_strength)

    # Free gpu memory
    del field_gradient

    if config.verbose:
        toc = time.perf_counter()
        print(f"Shadowing generation time: {toc - tic:0.4f} seconds")

    ############################### Final touches
    if config.verbose:
        tic = time.perf_counter()

    # Swap batch and channel dimensions
    field_channels = field_channels.permute(1, 0, 2, 3)

    # Resize to desired size
    output = v2.functional.resize(field_channels, config.field_size)

    # Convert to 16 bit numpy channels
    output = (output.round()).cpu().numpy().astype(np.uint16)

    # Count trees in the field
    tree_count = torch.sum(tree_boolmap.view(tree_boolmap.shape[0], -1), axis=-1)
    tree_count = tree_count.cpu().numpy().astype(np.uint16)

    # Adjust coordinates to match the output size and convert to numpy
    for i in range(len(tree_coordinates)):
        tree_coordinates[i] = tree_coordinates[i].to(torch.float32)
        tree_coordinates[i] = (tree_coordinates[i]/GAIN).cpu().numpy().astype(np.uint16)

    if config.verbose:
        toc = time.perf_counter()
        print(f"Final touches generation time: {toc - tic:0.4f} seconds")

    return output, tree_coordinates, tree_count
