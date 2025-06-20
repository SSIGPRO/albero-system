import os
import numpy as np
from argmanager import save_config_to_ini, get_parser_from_dict, update_config_from_args
from field_generator import generate_field
from postprocessing import postprocessing

# --- Default configuration dictionary ---
DEFAULT_CONFIG = {
    "verbose": True, # verbosity of the output
    "output_dir": "outputs", # directory to save the generated images
    "device": "cuda:0", # device to use for computation (e.g., "cuda:0", "cpu")
    ### FIELD SIZE:
    "batch_size": 1,
    "total_fields": 1,
    "field_size": (1000, 1000), # size of the output field in pixels
    "field_generation_zoom": 1, # zoom factor for the field generation
    ### TREE MAP:
    "treemap_size": (30, 30), # number of trees (rows, columns)
    "treemap_filter_size": 51, # size of the filter of the gaussian probability map
    "treemap_filter_sigma": 6,  # sigma of the filter of the gaussian probability map
    "treemap_noise_strength": 0.4, # strenght of final gaussian noise added to the map
    "treemap_sparse_remove_rate": 0.1, # rate of the sparse values to be minimized
    "treemap_sparse_add_rate": 0.05, # rate of the sparse values to be maximized
    "treemap_row_prune_rate": 0.05, # rate of the rows to be pruned
    "treemap_column_prune_rate": 0.05, # rate of the columns to be pruned
    "treemap_final_filter_size": 21, # size of the final filter
    "treemap_final_filter_sigma": 0.5, # sigma of the final filter
    ### TREE SPRITE SIZE AND POSITIONING:
    "tree_threshold": 0.3, # threshold for tree existence (1 -> no trees, 0 -> all trees) ### randomization from 0.3 to 0.6
    "tree_sprite_size": 80, # size of each tree sprite in pixels
    "tree_center_jitter": 5, # jitter of the tree center (in pixels)
    "tree_offset": -10, # offset (x and y of the same value) of the first top left tree (in pixels)
    "tree_alternate_offset": 20, # offset applied only to odd rows ### slight randomization
    "tree_xspace": 71, # x space between trees (in float pixels) from center to center ### slight randomization
    "tree_yspace": 71, # y space between trees (in float pixels) ### slight randomization
    "tree_steepness": 2.0, # steepness of the variation of tree size with respect to the probability map
    "tree_distribution_shift": 0.0, # shift of the variation of tree size with respect to the probability map
    ### TREE SHAPE:
    "treeshape_min_radius": 10, # minimum tree radius in pixels ### randomization 7 to 10
    "treeshape_max_radius": 14, # maximum tree radius ### randomization 12 to 14
    "treeshape_noise": 10, # noisyness of the tree shape (maximum amount in pixels)
    "treeshape_filter_size": 15, # size of the filter to smooth the tree shape
    "treeshape_filter_sigma": 1.0, # sigma of the filter to smooth the tree shape
    ### BACKGROUND:
    "bkg_overlay_filter_size": 25, # size of the filter to smooth the background overlay
    "bkg_overlay_filter_sigma": [5.0,], # sigma of the filter to smooth the background overlay
    "bkg_overlay_steepness": 20.0, # steepness of the stains produced as background overlay
    "bkg_noise_strength": 0.05, # strength of the noise added to the background
    "bkg_patches_steepness": 0.5, # steepness of the patches produced as background
    "bkg_patches_passes_min": 0, # minimum number of passes for the background patches
    "bkg_patches_passes_max": 10, # maximum number of passes for the background patches
    "bkg_patches_minmax_sizes": [(150, 400, 150, 400),], # maximum (width, height) of the background patches (in pixels)
    "bkg_patches_notrees_passes_min": 0, # minimum number of passes for the background patches (without overlapping trees)
    "bkg_patches_notrees_passes_max": 10, # maximum number of passes for the background patches (without overlapping trees)
    "bkg_patches_notrees_minmax_sizes": [(300, 700, 8, 20), (8, 20, 300, 700)], # maximum (width, height) of the background patches (in pixels) (without overlapping trees)
    ### SHADOWS AND LIGHTING:
    "shadow_direction": 2.0, # direction of the light source (in radians) ### randomization 0 to 2pi
    "shadow_length": 15, # length of the shadow (in pixels) ### randomization 0 to 20 (integer positive numbers)
    "shadow_iterations": 5, # iterations employed to produce the shadow
    "shadow_blending_strength": 0.1, # strength on the light/shadow effect on the tree sprite
    # COLORS_SET: in the order bkg, bkg_overlay, bkg_patches, bkg_patches_notrees, tree, shadows
    "color_sets": 
        [
        [#Img 4
            (128, 97, 79, 161), # background
            (78, 66, 61, 124), # overlay - macchie di colore gaussiano
            (95, 76, 69, 126), # patch rettangolari
            (56, 52, 52, 101), # patch senza alberi sovrapposti - delimitazione tra campi, strade
            (33, 36, 43, 103), # alberi
            (42, 42, 47, 92), # ombre
        ],
        [#Img 12
            (112, 84, 69, 142), # background
            (92, 73, 63, 132), # overlay - macchie di colore gaussiano
            (112, 84, 69, 142), # patch rettangolari
            (46, 44, 45, 87), # patch senza alberi sovrapposti - delimitazione tra campi, strade
            (28, 32, 37, 87), # alberi
            (35, 37, 41, 75), # ombre
        ], 
        [#Img 24
            (102, 76, 62, 130), # background
            (87, 68, 57, 123), # overlay - macchie di colore gaussiano
            (102, 76, 62, 130), # patch rettangolari
            (43, 41, 41, 84), # patch senza alberi sovrapposti - delimitazione tra campi, strade
            (26, 29, 35, 83), # alberi
            (33, 34, 37, 75), # ombre
        ], 
        [#Img 48
            (102, 78, 63, 133), # background
            (84, 67, 57, 124), # overlay - macchie di colore gaussiano
            (102, 78, 63, 133), # patch rettangolari
            (33, 36, 38, 91), # patch senza alberi sovrapposti - delimitazione tra campi, strade
            (25, 28, 34, 83), # alberi
            (42, 41, 41, 90), # ombre
        ],
        [#Img 68
            (104, 77, 65, 132), # background
            (63, 53, 52, 98), # overlay - macchie di colore gaussiano
            (77, 58, 54, 103), # patch rettangolari
            (44, 41, 44, 80), # patch senza alberi sovrapposti - delimitazione tra campi, strade
            (29, 32, 40, 84), # alberi
            (35, 36, 41, 77), # ombre
        ],
        [#Img 105
            (101, 79, 68, 132), # background
            (67, 63, 56, 131), # overlay - macchie di colore gaussiano
            (57, 56, 51, 122), # patch rettangolari
            (80, 69, 61, 125), # patch senza alberi sovrapposti - delimitazione tra campi, strade
            (28, 33, 38, 91), # alberi
            (39, 41, 43, 96), # ombre
        ],
        [#Img 150
            (127, 99, 83, 168), # background
            (83, 74, 67, 144), # overlay - macchie di colore gaussiano
            (127, 99, 83, 168), # patch rettangolari
            (102, 86, 76, 149), # patch senza alberi sovrapposti - delimitazione tra campi, strade
            (36, 41, 48, 114), # alberi
            (47, 50, 53, 117), # ombre
        ],
        [#Img 172
            (102, 81, 66, 144), # background
            (67, 61, 54, 129), # overlay - macchie di colore gaussiano
            (52, 51, 48, 118), # patch rettangolari
            (35, 38, 40, 95), # patch senza alberi sovrapposti - delimitazione tra campi, strade
            (25, 30, 37, 71), # alberi
            (46, 40, 41, 90), # ombre
        ],
        [#Img 268
            (74, 69, 63, 146), # background
            (61, 62, 59, 147), # overlay - macchie di colore gaussiano
            (41, 49, 51, 135), # patch rettangolari
            (33, 44, 48, 129), # patch senza alberi sovrapposti - delimitazione tra campi, strade
            (25, 34, 43, 78), # alberi
            (46, 40, 41, 90), # ombre
        ],
        [#Img 347
            (119, 89, 72, 149), # background
            (99, 79, 67, 137), # overlay - macchie di colore gaussiano
            (59, 55, 52, 114), # patch rettangolari
            (76, 65, 58, 124), # patch senza alberi sovrapposti - delimitazione tra campi, strade
            (34, 37, 43, 98), # alberi
            (44, 44, 47, 97), # ombre
        ],
        [#Img 386
            (128, 101, 84, 174), # background
            (64, 64, 61, 148), # overlay - macchie di colore gaussiano
            (44, 53, 54, 148), # patch rettangolari
            (86, 77, 69, 155), # patch senza alberi sovrapposti - delimitazione tra campi, strade
            (22, 31, 41, 87), # alberi
            (31, 41, 47, 131), # ombre
        ],
        [#Img 394
            (135, 101, 81, 168), # background
            (87, 72, 65, 132), # overlay - macchie di colore gaussiano
            (95, 73, 64, 125), # patch rettangolari
            (95, 73, 64, 125), # patch senza alberi sovrapposti - delimitazione tra campi, strade
            (35, 38, 46, 103), # alberi
            (47, 47, 50, 104), # ombre
        ],
        [#Img 506
            (95, 71, 59, 122), # background
            (48, 44, 42, 92), # overlay - macchie di colore gaussiano
            (74, 57, 50, 101), # patch rettangolari
            (66, 55, 50, 106), # patch senza alberi sovrapposti - delimitazione tra campi, strade
            (26, 30, 35, 84), # alberi
            (35, 35, 38, 82), # ombre
        ],
        ],
    "color_randomization_strength": 0.05, # relative random change for each color channel
    ### POSTPROCESSING:
    "tiles_per_side": 1, # number of tiles per side when glueing together the tiles (e.g., 2 -> 4 tiles, 3 -> 9 tiles, etc.)
    "view_rotation_deg": 20, # rotation of the view (in degrees) ### randomization from 0 to 360
    "tile_size": 640, # size of the tiles (in pixels)
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
    if config["verbose"]:
        print("Configuration:")
        for k, v in config.items():
            print(f"{k}: {v}")

    # Call main function
    outputs_list = []
    coordinates_list = []
    count_list = []
    for _ in range(config["total_fields"]//config["batch_size"]):
        outputs, coordinates, count = generate_field(**config)
        outputs_list.append(outputs)
        coordinates_list += coordinates
        count_list += count

    outputs = np.concatenate(outputs_list)
    coordinates = coordinates_list
    count = count_list

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