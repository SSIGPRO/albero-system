import torch
from torchvision.transforms import v2
import numpy as np
import time
from base import gaussian_filter, normalize_tensor, normalize_positive_tensor
from types import SimpleNamespace
from generators import generate_probability_maps,\
                       generate_tree_sprites,\
                       prob_to_treesize,\
                       generate_field_mask,\
                       project_shadow,\
                       compute_gradient_map,\
                       apply_lighting,\
                       bool_tensor_to_coords

def generate_field(**kwargs):
    config = SimpleNamespace(**kwargs)

    ############################### Background color + overlay
    if config.verbose:
        tic = time.perf_counter()

    GAIN = config.field_generation_zoom
    CANVAS_SIZE = config.field_size[0]*GAIN, config.field_size[1]*GAIN
    
    # generate background color
    canvas_shape = (config.batch_size,
                    CANVAS_SIZE[0],
                    CANVAS_SIZE[1])
    
    field_channels = torch.stack([
        torch.full(canvas_shape, config.color_bkg[0], device=config.device, dtype=torch.float32),
        torch.full(canvas_shape, config.color_bkg[1], device=config.device, dtype=torch.float32),
        torch.full(canvas_shape, config.color_bkg[2], device=config.device, dtype=torch.float32),
        torch.full(canvas_shape, config.color_bkg[3], device=config.device, dtype=torch.float32)
    ])

    # apply overlay to background
    overlay_margin = 1
    background_mask2 = torch.randn_like(field_channels[0])
    background_mask2[..., :, :overlay_margin] = -1
    background_mask2[..., :overlay_margin, :] = -1
    background_mask2[..., :, -overlay_margin:] = -1
    background_mask2[..., -overlay_margin:, :] = -1
    background_mask2 = torch.sigmoid(gaussian_filter(background_mask2, 25, 8.0)*100-1)
    for i in range(4):
        field_channels[i] = field_channels[i]*(1-background_mask2) + config.color_bkg_overlay[i]*background_mask2

    # free gpu memory
    del background_mask2

    if config.verbose:
        toc = time.perf_counter()
        print(f"Background color generation time: {toc - tic:0.4f} seconds")

    ############################### Apply stains to background
    if config.verbose:
        tic = time.perf_counter()

    # Generate background stains map
    bkg_stain_map_shape = (config.batch_size, config.treemap_size[0], config.treemap_size[1]*3)
    bkg_stain_map = torch.rand(bkg_stain_map_shape, device=config.device)*(config.bkg_stain_strength-config.bkg_stain_strength_min)+config.bkg_stain_strength_min

    # Generate background stains sprites
    bkg_stain_sprites, _, _ = generate_tree_sprites(bkg_stain_map,
                                                    tree_sprite_size=config.bkg_stain_pixel_size,
                                                    tree_size=5,
                                                    max_tree_size=10,
                                                    center_jitter=[config.bkg_stain_center_jitter]*2)

    # Free gpu memory
    del bkg_stain_map

    # Generate field background stains mask
    background_mask = generate_field_mask(bkg_stain_sprites,
                                          CANVAS_SIZE,
                                          distance=(config.tree_xspace*GAIN//3, config.tree_yspace*GAIN),
                                          offset=config.bkg_stain_offset*GAIN)

    # Free gpu memory
    del bkg_stain_sprites

    # apply stains to background
    for i in range(4):
        field_channels[i] = field_channels[i]*(1-background_mask) + config.color_bkg_overlay[i]*background_mask

    # free gpu memory
    del background_mask

    if config.verbose:
        toc = time.perf_counter()
        print(f"Background stains generation time: {toc - tic:0.4f} seconds")

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
                                         column_prune_rate=config.treemap_column_prune_rate,
                                         final_filter_size=config.treemap_final_filter_size,
                                         final_filter_sigma=config.treemap_final_filter_sigma,
                                         device=config.device)
    
    # Generate tree size map and number of trees labels
    treesize_map, tree_boolmap, tree_count = prob_to_treesize(prob_map, 
                                                              threshold=config.tree_threshold,
                                                              steepness=config.tree_steepness,
                                                              distribution_shift=config.tree_distribution_shift)
    
    # Free gpu memory
    del prob_map

    if config.verbose:
        toc = time.perf_counter()
        print(f"Field of trees probability map generation time: {toc - tic:0.4f} seconds")

    if config.verbose:
        tic = time.perf_counter()

    # Generate tree sprites
    tree_sprites, tree_offsets, tree_sizes = generate_tree_sprites(treesize_map=treesize_map,
                                                                   tree_size=config.treeshape_size*GAIN,
                                                                   max_tree_size=config.treeshape_max_size*GAIN,
                                                                   tree_sprite_size=config.tree_sprite_size*GAIN,
                                                                   center_jitter=(config.tree_center_jitter*GAIN, config.tree_center_jitter*GAIN),
                                                                   noise_level=config.treeshape_size*GAIN,
                                                                   filter_size=config.treeshape_filter_size,
                                                                   filter_sigma=config.treeshape_filter_sigma,)
    
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

    ############################### Generate coordinates and sizes
    if config.verbose:
        tic = time.perf_counter()

    # Convert treebool_map to coordinate labels
    constant_offset = [config.tree_offset*GAIN + config.tree_sprite_size*GAIN]*2
    tree_coordinates = bool_tensor_to_coords(tree_boolmap,
                                             tree_offsets,
                                             constant_offset,
                                             tree_sizes,
                                             config.tree_xspace*GAIN,
                                             config.tree_yspace*GAIN,)
    
    if config.verbose:
        toc = time.perf_counter()
        print(f"Field of trees coordinates generation time: {toc - tic:0.4f} seconds")

    ############################### Generate and apply projected shadows
    if config.verbose:
        tic = time.perf_counter()

    # Generate shadows mask
    field_shadows_mask = project_shadow(field_tree_mask,
                                        config.shadow_iterations,
                                        config.shadow_length*GAIN,
                                        config.shadow_direction)

    # apply projected shadows to backgroud
    for i in range(4):
        field_channels[i] = field_channels[i]*(1-field_shadows_mask) + config.color_tree_shadow[i]*field_shadows_mask

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
        field_channels[i] = field_channels[i]*(1-field_tree_mask) + config.color_tree[i]*field_tree_mask

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

    # Apply filter
    field_channels = gaussian_filter(field_channels, 9, 1.0)

    # Swap batch and channel dimensions
    field_channels = field_channels.permute(1, 0, 2, 3)

    # Resize to desired size
    output = v2.functional.resize(field_channels, config.field_size)

    # Convert to 16 bit numpy channels
    output = (output.round()).cpu().numpy().astype(np.uint16)
    tree_count = tree_count.cpu().numpy().astype(np.uint16)

    # Adjust coordinates to match the output size and convert to numpy
    for i in range(len(tree_coordinates)):
        tree_coordinates[i] = tree_coordinates[i].to(torch.float32) 
        tree_coordinates[i] = (tree_coordinates[i]/GAIN).cpu().numpy().astype(np.uint16)

    if config.verbose:
        toc = time.perf_counter()
        print(f"Final touches generation time: {toc - tic:0.4f} seconds")

    return output, tree_coordinates, tree_count