import os
import numpy as np
from argmanager import save_config_to_ini, get_parser_from_dict, update_config_from_args
from field_generator import generate_field
from postprocessing import postprocessing

# --- Default configuration dictionary ---
DEFAULT_CONFIG = {
    "batch_size": 32,
    "output_size": (600, 600), # size of the output image ### modify depending on the model (600x600 might be too big)
    "output_dir": "outputs", # directory to save the generated images
    "device": "cuda:0", # device to use for computation (e.g., "cuda:0", "cpu")
    "treemap_size": (52, 104), # number of trees (rows, columns)
    "treemap_filter_size": 51, # size of the filter of the gaussian probability map
    "treemap_filter_sigma": 6,  # sigma of the filter of the gaussian probability map ### maybe slight randomization
    "treemap_noise_strength": 0.1, # strenght of final gaussian noise added to the map ### maybe slight randomization
    "treemap_sparse_remove_rate": 0.05, # rate of the sparse values to be minimized ### maybe slight randomization
    "treemap_sparse_add_rate": 0.05, # rate of the sparse values to be maximized ### maybe slight randomization
    "treemap_column_prune_rate": 0.01, # rate of the columns to be pruned ### maybe slight randomization
    "treemap_final_filter_size": 21, # size of the final filter
    "treemap_final_filter_sigma": 0.5, # sigma of the final filter
    "tree_threshold": 0.4, # threshold for tree existence (1 -> no trees, 0 -> all trees) ### randomization from 0 to 1
    "tree_pixel_size": 40, # size of each tree sprite in pixels
    "tree_center_jitter": 5, # jitter of the tree center (in pixels) ### maybe slight randomization (integer positive numbers)
    "tree_xspace": 15, # x space between trees (in pixels)
    "tree_yspace": 30, # y space between trees (in pixels)
    "treesize_steepness": 4.0, # steepness of the variation of tree size with respect to the probability map ### maybe slight randomization
    "treesize_distribution_shift": 0.0, # shift of the variation of tree size with respect to the probability map ### maybe slight randomization (-0.5 to 0.5)
    "treesize_gain": 0.6, # tree size with respect to the sprite size ### maybe slight randomization
    "treesize_max": 0.8, # maximum radius of the tree with respect to the tree sprite
    "treesize_noise": 1.5, # noisyness of the tree shape
    "bkg_stain_pixel_size": 20, # size of each background stains in pixels
    "bkg_stain_center_jitter": 5, # jitter of the background stains center (in pixels)
    "bkg_stain_strength_min": 0.2, # minimum strength of the background stains
    "bkg_stain_strength": 0.8, # maximum strength of the background stains
    "bkg_stain_offset": 15, # x and y offset of the background stains (in pixels)
    "bkg_noise_strength": 0.1, # strength of the noise added to the background
    "field_generated_size": (1610, 1610), # size (no margins included) of the generated field
    "field_generated_margin": 30, # margins of the generated field
    "shadow_direction": 2.0, # direction of the light source (in radians) ### randomization 0 to 2pi
    "shadow_length": 15, # length of the shadow (in pixels) ### randomization 0 to 20 (integer positive numbers)
    "shadow_iterations": 5, # iterations employed to produce the shadow
    "shadow_blending_strength": 0.1, # strength on the light/shadow effect on the tree sprite
    "color_bkg": (700, 520, 420, 800), # color of the background(R, G, B, NIF) ### maybe slight randomization + verification with the actual images
    "color_bkg_stain": (400, 280, 320, 800), # color of the background stains (R, G, B, NIF) ### maybe slight randomization + verification with the actual images
    "color_bkg_overlay": (550, 430, 420, 700), # color of the background overlays (R, G, B, NIF) ### maybe slight randomization + verification with the actual images
    "color_tree": (220, 280, 300, 1150), # color of the trees (R, G, B, NIF) ### maybe slight randomization + verification with the actual images
    "color_tree_shadow": (80, 120, 130, 550), # color of the tree shadows(R, G, B, NIF) ### maybe slight randomization + verification with the actual images
    ### POSTPROCESSING:
    "tiles_per_side": 4, # number of tiles per side when glueing together the tiles (e.g., 2 -> 4 tiles, 3 -> 9 tiles, etc.)
    "view_rotation_deg": 60, # rotation of the view (in degrees) ### randomization from 0 to 360
    "tile_size": 256, # size of the tiles (in pixels)
}

def main():
    # Parse command-line arguments
    parser = get_parser_from_dict(DEFAULT_CONFIG)
    args = parser.parse_args()

    # Update config from args
    config = DEFAULT_CONFIG.copy()
    config = update_config_from_args(config, args)

    # Save final config to .ini file
    os.makedirs(config["output_dir"], exist_ok=True)
    save_config_to_ini(config, os.path.join(config["output_dir"], "config.ini"))

    # Print configuration
    print("Configuration:")
    for k, v in config.items():
        print(f"{k}: {v}")

    # Call main function
    outputs, coordinates, count = generate_field(**config)
    outputs, coordinates, count = postprocessing(outputs, coordinates, **config)

    # Save outputs and labels
    output_path = os.path.join("outputs", f"output.npy")
    coordinates_path = os.path.join("outputs", f"coordinates.npz")
    count_path = os.path.join("outputs", f"count.npy")
    np.save(output_path, outputs)
    np.savez(coordinates_path, **{f'coords{i}': t for i, t in enumerate(coordinates)})
    np.save(count_path, count)

if __name__ == "__main__":
    main()