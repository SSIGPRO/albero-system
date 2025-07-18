"""Configuration constants for the field generator."""

from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "verbose": True,  # verbosity of the output
    "output_dir": "./outputs",  # directory to save the generated images
    "device": "cuda:0",  # device to use for computation (e.g., "cuda:0", "cpu")
    ### FIELD SIZE:
    "batch_size": 1,
    "total_fields": 1,
    "field_size": (1000, 1000),  # size of the output field in pixels
    "field_generation_zoom": 1,  # zoom factor for the field generation
    ### TREE MAP:
    "treemap_size": (30, 30),  # number of trees (rows, columns)
    "treemap_filter_size": 51,  # size of the filter of the gaussian probability map
    "treemap_filter_sigma": 6,  # sigma of the filter of the gaussian probability map
    "treemap_noise_strength": 0.4,  # strength of final gaussian noise added to the map
    "treemap_sparse_remove_rate": 0.1,  # rate of the sparse values to be minimized
    "treemap_sparse_add_rate": 0.05,  # rate of the sparse values to be maximized
    "treemap_row_prune_rate": 0.05,  # rate of the rows to be pruned
    "treemap_column_prune_rate": 0.05,  # rate of the columns to be pruned
    "treemap_final_filter_size": 21,  # size of the final filter
    "treemap_final_filter_sigma": 0.5,  # sigma of the final filter
    ### TREE SPRITE SIZE AND POSITIONING:
    "tree_threshold": 0.3,  # threshold for tree existence (1 -> no trees, 0 -> all trees) ### randomization from 0.3 to 0.6
    "tree_sprite_size": 80,  # size of each tree sprite in pixels
    "tree_center_jitter": 5,  # jitter of the tree center (in pixels)
    "tree_offset": -10,  # offset (x and y of the same value) of the first top left tree (in pixels)
    "tree_alternate_offset": 20,  # offset applied only to odd rows ### slight randomization
    "tree_xspace": 71,  # x space between trees (in float pixels) from center to center ### slight randomization
    "tree_yspace": 71,  # y space between trees (in float pixels) ### slight randomization
    "tree_steepness": 2.0,  # steepness of the variation of tree size with respect to the probability map
    "tree_distribution_shift": 0.0,  # shift of the variation of tree size with respect to the probability map
    ### TREE SHAPE:
    "treeshape_min_radius": 10,  # minimum tree radius in pixels ### randomization 7 to 10
    "treeshape_max_radius": 14,  # maximum tree radius ### randomization 12 to 14
    "treeshape_noise": 10,  # noisiness of the tree shape (maximum amount in pixels)
    "treeshape_filter_size": 15,  # size of the filter to smooth the tree shape
    "treeshape_filter_sigma": 1.0,  # sigma of the filter to smooth the tree shape
    ### BACKGROUND:
    "bkg_overlay_filter_size": 25,  # size of the filter to smooth the background overlay
    "bkg_overlay_filter_sigma": [5.0],  # sigma of the filter to smooth the background overlay
    "bkg_overlay_steepness": 20.0,  # steepness of the stains produced as background overlay
    "bkg_noise_strength": 0.05,  # strength of the noise added to the background
    "bkg_patches_steepness": 0.5,  # steepness of the patches produced as background
    "bkg_patches_passes_min": 0,  # minimum number of passes for the background patches
    "bkg_patches_passes_max": 10,  # maximum number of passes for the background patches
    "bkg_patches_minmax_sizes": [(150, 400, 150, 400)],  # maximum (width, height) of the background patches (in pixels)
    "bkg_patches_notrees_passes_min": 0,  # minimum number of passes for the background patches (without overlapping trees)
    "bkg_patches_notrees_passes_max": 10,  # maximum number of passes for the background patches (without overlapping trees)
    "bkg_patches_notrees_minmax_sizes": [
        (300, 700, 8, 20),
        (8, 20, 300, 700),
    ],  # maximum (width, height) of the background patches (in pixels) (without overlapping trees)
    ### SHADOWS AND LIGHTING:
    "shadow_direction": 2.0,  # direction of the light source (in radians) ### randomization 0 to 2pi
    "shadow_length": 15,  # length of the shadow (in pixels) ### randomization 0 to 20 (integer positive numbers)
    "shadow_iterations": 5,  # iterations employed to produce the shadow
    "shadow_blending_strength": 0.1,  # strength on the light/shadow effect on the tree sprite
    # COLORS_SET: in the order bkg, bkg_overlay, bkg_patches, bkg_patches_notrees, tree, shadows
    "color_sets": [
        [  # Image 4
            (128, 97, 79, 161),  # background
            (78, 66, 61, 124),  # overlay - gaussian color patches
            (95, 76, 69, 126),  # rectangular patches
            (56, 52, 52, 101),  # patches without overlapping trees - field boundaries, roads
            (33, 36, 43, 103),  # trees
            (42, 42, 47, 92),  # shadows
        ],
        [  # Image 12
            (112, 84, 69, 142),  # background
            (92, 73, 63, 132),  # overlay - gaussian color patches
            (112, 84, 69, 142),  # rectangular patches
            (46, 44, 45, 87),  # patches without overlapping trees - field boundaries, roads
            (28, 32, 37, 87),  # trees
            (35, 37, 41, 75),  # shadows
        ],
        [  # Image 24
            (102, 76, 62, 130),  # background
            (87, 68, 57, 123),  # overlay - gaussian color patches
            (102, 76, 62, 130),  # rectangular patches
            (43, 41, 41, 84),  # patches without overlapping trees - field boundaries, roads
            (26, 29, 35, 83),  # trees
            (33, 34, 37, 75),  # shadows
        ],
        [  # Image 48
            (102, 78, 63, 133),  # background
            (84, 67, 57, 124),  # overlay - gaussian color patches
            (102, 78, 63, 133),  # rectangular patches
            (33, 36, 38, 91),  # patches without overlapping trees - field boundaries, roads
            (25, 28, 34, 83),  # trees
            (42, 41, 41, 90),  # shadows
        ],
        [  # Image 68
            (104, 77, 65, 132),  # background
            (63, 53, 52, 98),  # overlay - gaussian color patches
            (77, 58, 54, 103),  # rectangular patches
            (44, 41, 44, 80),  # patches without overlapping trees - field boundaries, roads
            (29, 32, 40, 84),  # trees
            (35, 36, 41, 77),  # shadows
        ],
        [  # Image 105
            (101, 79, 68, 132),  # background
            (67, 63, 56, 131),  # overlay - gaussian color patches
            (57, 56, 51, 122),  # rectangular patches
            (80, 69, 61, 125),  # patches without overlapping trees - field boundaries, roads
            (28, 33, 38, 91),  # trees
            (39, 41, 43, 96),  # shadows
        ],
        [  # Image 150
            (127, 99, 83, 168),  # background
            (83, 74, 67, 144),  # overlay - gaussian color patches
            (127, 99, 83, 168),  # rectangular patches
            (102, 86, 76, 149),  # patches without overlapping trees - field boundaries, roads
            (36, 41, 48, 114),  # trees
            (47, 50, 53, 117),  # shadows
        ],
        [  # Image 172
            (102, 81, 66, 144),  # background
            (67, 61, 54, 129),  # overlay - gaussian color patches
            (52, 51, 48, 118),  # rectangular patches
            (35, 38, 40, 95),  # patches without overlapping trees - field boundaries, roads
            (25, 30, 37, 71),  # trees
            (46, 40, 41, 90),  # shadows
        ],
        [  # Image 268
            (74, 69, 63, 146),  # background
            (61, 62, 59, 147),  # overlay - gaussian color patches
            (41, 49, 51, 135),  # rectangular patches
            (33, 44, 48, 129),  # patches without overlapping trees - field boundaries, roads
            (25, 34, 43, 78),  # trees
            (46, 40, 41, 90),  # shadows
        ],
        [  # Image 347
            (119, 89, 72, 149),  # background
            (99, 79, 67, 137),  # overlay - gaussian color patches
            (59, 55, 52, 114),  # rectangular patches
            (76, 65, 58, 124),  # patches without overlapping trees - field boundaries, roads
            (34, 37, 43, 98),  # trees
            (44, 44, 47, 97),  # shadows
        ],
        [  # Image 386
            (128, 101, 84, 174),  # background
            (64, 64, 61, 148),  # overlay - gaussian color patches
            (44, 53, 54, 148),  # rectangular patches
            (86, 77, 69, 155),  # patches without overlapping trees - field boundaries, roads
            (22, 31, 41, 87),  # trees
            (31, 41, 47, 131),  # shadows
        ],
        [  # Image 394
            (135, 101, 81, 168),  # background
            (87, 72, 65, 132),  # overlay - gaussian color patches
            (95, 73, 64, 125),  # rectangular patches
            (95, 73, 64, 125),  # patches without overlapping trees - field boundaries, roads
            (35, 38, 46, 103),  # trees
            (47, 47, 50, 104),  # shadows
        ],
        [  # Image 506
            (95, 71, 59, 122),  # background
            (48, 44, 42, 92),  # overlay - gaussian color patches
            (74, 57, 50, 101),  # rectangular patches
            (66, 55, 50, 106),  # patches without overlapping trees - field boundaries, roads
            (26, 30, 35, 84),  # trees
            (35, 35, 38, 82),  # shadows
        ],
    ],
    "color_randomization_strength": 0.05,  # relative random change for each color channel
    ### POSTPROCESSING:
    "tiles_per_side": 1,  # number of tiles per side when gluing together the tiles (e.g., 2 -> 4 tiles, 3 -> 9 tiles, etc.)
    "view_rotation_deg": 20,  # rotation of the view (in degrees) ### randomization from 0 to 360
    "tile_size": 640,  # size of the tiles (in pixels)
}
