import numpy as np
import time
from types import SimpleNamespace
from src.glue_n_split import glue_together,\
                         rotate_n_crop,\
                         rotate_coords,\
                         crop_coordinates,\
                         split_n_crop,\
                         coords_to_treecount


def postprocessing(batch, coordinates, **kwargs):
    config = SimpleNamespace(**kwargs)

    if config.verbose:
        tic = time.perf_counter()

    # Glue together multiple generated tiles
    newimage, newcoords = glue_together(batch, coordinates, config.tiles_per_side)
    # Rotate and crop the glued coordinates and the image
    newcoords = rotate_coords(newcoords, (newimage.shape[-2]//2, newimage.shape[-1]//2), config.view_rotation_deg)
    newimage, crop = rotate_n_crop(newimage, config.view_rotation_deg)
    newcoords = crop_coordinates(newcoords, crop, crop, newimage.shape[-2], newimage.shape[-1])

    # Split the image into tiles and crop the coordinates
    newimage, newcoords = split_n_crop(newimage, newcoords, config.tile_size)

    # Convert coordinates to tree counts
    treecount = coords_to_treecount(newcoords)

    if config.verbose:
        toc = time.perf_counter()
        print(f"Postprocessing time: {toc - tic:0.4f} seconds")

    newimage = newimage.astype(np.uint16)
    for i in range(len(newcoords)):
        newcoords[i] = newcoords[i].astype(np.uint16)
    treecount = treecount.astype(np.uint16)

    return newimage, newcoords, treecount
