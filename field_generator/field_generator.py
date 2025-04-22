import torch
from torchvision.transforms import v2
import numpy as np
import time
from base import gaussian_filter, normalize_tensor, normalize_positive_tensor
from types import SimpleNamespace
from generators import generate_probability_maps,\
                       generate_tree_sprites,\
                       prob_to_treesize,\
                       generate_tree_sprites,\
                       generate_field_mask,\
                       project_shadow,\
                       compute_gradient_map,\
                       apply_lighting,\
                       bool_tensor_to_coords

def generate_field(**kwargs):
    config = SimpleNamespace(**kwargs)

    ############################### Background color + overlay
    tic = time.perf_counter()

    # generate background color
    canvas_shape = (config.batch_size,
                    config.field_generated_size[0] + 2*config.field_generated_margin,
                    config.field_generated_size[1] + 2*config.field_generated_margin)
    
    field_channels = torch.stack([
        torch.full(canvas_shape, config.color_bkg[0], device=config.device, dtype=torch.float32),
        torch.full(canvas_shape, config.color_bkg[1], device=config.device, dtype=torch.float32),
        torch.full(canvas_shape, config.color_bkg[2], device=config.device, dtype=torch.float32),
        torch.full(canvas_shape, config.color_bkg[3], device=config.device, dtype=torch.float32)
    ])

    # apply overlay 2 to background
    background_mask2 = torch.randn_like(field_channels[0])
    background_mask2 = torch.sigmoid(gaussian_filter(background_mask2, 101, 16)*200-1)
    for i in range(4):
        field_channels[i] = field_channels[i]*(1-background_mask2) + config.color_bkg_overlay[i]*background_mask2

    # free gpu memory
    del background_mask2

    toc = time.perf_counter()
    print(f"Background color generation time: {toc - tic:0.4f} seconds")

    ############################### Apply stains to background
    tic = time.perf_counter()

    # Generate background stains map
    bkg_stain_map_shape = (config.batch_size, config.treemap_size[0], config.treemap_size[1]*3)
    bkg_stain_map = torch.rand(bkg_stain_map_shape, device=config.device)*(config.bkg_stain_strength-config.bkg_stain_strength_min)+config.bkg_stain_strength_min

    # Generate background stains sprites
    bkg_stain_sprites, _ = generate_tree_sprites(bkg_stain_map, config.bkg_stain_pixel_size, [config.bkg_stain_center_jitter]*2)

    # Free gpu memory
    del bkg_stain_map

    # Generate field background stains mask
    background_mask = generate_field_mask(bkg_stain_sprites, config.field_generated_size, distance=(config.tree_xspace//3, config.tree_yspace))
    margin_tuple = (config.field_generated_margin+config.bkg_stain_offset,
                    config.field_generated_margin-config.bkg_stain_offset,
                    config.field_generated_margin+config.bkg_stain_offset,
                    config.field_generated_margin-config.bkg_stain_offset)
    background_mask = torch.nn.functional.pad(background_mask, margin_tuple, "constant")

    # Free gpu memory
    del bkg_stain_sprites

    # apply overlay 1 to background
    for i in range(4):
        field_channels[i] = field_channels[i]*(1-background_mask) + config.color_bkg_overlay[i]*background_mask

    # free gpu memory
    del background_mask
    
    toc = time.perf_counter()
    print(f"Background stains generation time: {toc - tic:0.4f} seconds")

    ############################### Apply noise to background
    tic = time.perf_counter()

    # apply noise to bkg
    noise = torch.randn_like(field_channels[0])*config.bkg_noise_strength
    field_channels *= (1+noise)

    # free gpu memory
    del noise

    toc = time.perf_counter()
    print(f"Background noise generation time: {toc - tic:0.4f} seconds")

    ############################### Generate field of trees
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
                                                              steepness=config.treesize_steepness,
                                                              distribution_shift=config.treesize_distribution_shift)
    
    # Free gpu memory
    del prob_map

    toc = time.perf_counter()
    print(f"Field of trees probability map generation time: {toc - tic:0.4f} seconds")

    tic = time.perf_counter()

    # Generate tree sprites
    tree_sprites, tree_offsets = generate_tree_sprites(treesize_map=treesize_map,
                                                       tree_pixel_size=config.tree_pixel_size,
                                                       center_jitter=(config.tree_center_jitter, config.tree_center_jitter),
                                                       treesize_gain=config.treesize_gain,
                                                       max_radius=config.treesize_max,
                                                       noise_level=config.treesize_noise)
    
    # Free gpu memory
    del treesize_map

    # Generate field tree mask
    field_tree_mask = generate_field_mask(tree_sprites, config.field_generated_size, distance=(config.tree_xspace, config.tree_yspace))
    margin_tuple = tuple([config.field_generated_margin]*4)
    field_tree_mask = torch.nn.functional.pad(field_tree_mask, margin_tuple, "constant")

    # Free gpu memory
    del tree_sprites

    toc = time.perf_counter()
    print(f"Field of trees generation time: {toc - tic:0.4f} seconds")

    ############################### Generate coordinates
    tic = time.perf_counter()

    # Convert treebool_map to coordinate labels
    constant_offset = [config.field_generated_margin + config.tree_pixel_size]*2
    tree_coordinates = bool_tensor_to_coords(tree_boolmap,
                                             tree_offsets,
                                             constant_offset,
                                             config.tree_xspace,
                                             config.tree_yspace,)
    
    toc = time.perf_counter()
    print(f"Field of trees coordinates generation time: {toc - tic:0.4f} seconds")

    ############################### Generate and apply projected shadows
    tic = time.perf_counter()

    # Generate shadows mask
    field_shadows_mask = project_shadow(field_tree_mask,
                                        config.shadow_iterations,
                                        config.shadow_length,
                                        config.shadow_direction)

    # apply projected shadows to backgroud
    for i in range(4):
        field_channels[i] = field_channels[i]*(1-field_shadows_mask) + config.color_tree_shadow[i]*field_shadows_mask

    toc = time.perf_counter()
    print(f"Projected shadows generation time: {toc - tic:0.4f} seconds")

    ############################### Background gaussian blur
    tic = time.perf_counter()

    # smooth background
    field_channels = gaussian_filter(field_channels, 9, 2.0)

    toc = time.perf_counter()
    print(f"Background gaussian blur time: {toc - tic:0.4f} seconds")

    ############################### Put trees on the field
    tic = time.perf_counter()

    # Put trees on the field
    for i in range(4):
        field_channels[i] = field_channels[i]*(1-field_tree_mask) + config.color_tree[i]*field_tree_mask

    toc = time.perf_counter()
    print(f"Trees generation time: {toc - tic:0.4f} seconds")

    ############################### Apply shadowing on the trees
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

    toc = time.perf_counter()
    print(f"Shadowing generation time: {toc - tic:0.4f} seconds")

    ############################### Final touches
    tic = time.perf_counter()

    # Apply filter
    field_channels = gaussian_filter(field_channels, 9, 1.0)

    # Swap batch and channel dimensions
    field_channels = field_channels.permute(1, 0, 2, 3)

    # Resize to desired size
    output = v2.functional.resize(field_channels, config.output_size)

    # Convert to 16 bit numpy channels
    output = (output.round()).cpu().numpy().astype(np.uint16)
    tree_count = tree_count.cpu().numpy().astype(np.uint16)

    # Adjust coordinates to match the output size and convert to numpy
    for i in range(len(tree_coordinates)):
        tree_coordinates[i] = tree_coordinates[i].to(torch.float32) 
        tree_coordinates[i][:, 0] = tree_coordinates[i][:, 0]*float(config.output_size[0])/float(config.field_generated_size[0]+2*config.field_generated_margin)
        tree_coordinates[i][:, 1] = tree_coordinates[i][:, 1]*float(config.output_size[1])/float(config.field_generated_size[1]+2*config.field_generated_margin)
        tree_coordinates[i] = tree_coordinates[i].cpu().numpy().astype(np.uint16)

    toc = time.perf_counter()
    print(f"Final touches generation time: {toc - tic:0.4f} seconds")

    return output, tree_coordinates, tree_count