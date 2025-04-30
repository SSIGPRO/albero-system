import numpy as np
import torch
import math
from torchvision.transforms import v2

def coords_npz_to_list(npz_coordinates):
    coords = []
    for i in range(len(npz_coordinates)):
        coords.append(npz_coordinates[f"coords{i}"].astype(int))
    return coords

def _tile_nxn(images, tile_n):
    """
    images: (tile_n**2 * B, C, H, W)
    tile_n: int, e.g., 2, 3, 4, ...
    Returns: (B, C, tile_n * H, tile_n * W)
    """
    total, C, H, W = images.shape
    B = total // (tile_n ** 2)

    # Step 1: reshape into (B, tile_n, tile_n, C, H, W)
    images = images.reshape(B, tile_n, tile_n, C, H, W)

    # Step 2: transpose to move tiling into spatial dimensions
    images = images.transpose(0, 3, 1, 4, 2, 5)  # (B, C, tile_n, H, tile_n, W)

    # Step 3: merge tiling into height and width
    images = images.reshape(B, C, tile_n * H, tile_n * W)

    return images

def glue_together(image, coords, tile_number):
    newimage = _tile_nxn(image, tile_number)
    newcoords = []
    for k in range(len(coords)//(tile_number**2)):
        newcoor_list = []
        for i in range(tile_number):
            for j in range(tile_number):
                coor = coords[k*tile_number**2+i*tile_number+j]
                newcoor_list.append(coor + np.array([j*image.shape[-2], i*image.shape[-1]]))
        newcoords.append(np.concatenate(newcoor_list, axis=0))
    return newimage, newcoords

def rotate_n_crop(image, angle_deg):
    image = torch.tensor(image)
    image = v2.functional.rotate(image, angle_deg)
    angle_rad = math.radians(angle_deg)
    radius = image.shape[-1]//2
    angle_rad_rest = angle_rad % (math.pi/2)
    crop_factor = (math.sqrt(2) - 1/math.cos(math.pi/4-angle_rad_rest))*math.cos(math.pi/4)
    crop = math.ceil(crop_factor * radius)
    image = v2.functional.crop(image, crop, crop, image.shape[-2]-crop*2, image.shape[-1]-crop*2)
    return np.array(image), crop

def rotate_coords(coords, origin, angle_deg):
    newcoords = []
    for coor in coords:
        newcoor = np.zeros_like(coor)
        angle_rad = math.radians(angle_deg)
        dx, dy = coor[:, 0] - origin[0], coor[:, 1] - origin[1]
        newcoor[:, 0] = origin[0] + dx * math.cos(-angle_rad) - dy * math.sin(-angle_rad)
        newcoor[:, 1] = origin[1] + dx * math.sin(-angle_rad) + dy * math.cos(-angle_rad)
        newcoords.append(newcoor)
    return newcoords

def crop_coordinates(coords, crop_left, crop_top, crop_width, crop_height):
    newcoords = []
    for coor in coords:
        newcoor = np.zeros_like(coor)
        newcoor[:, 0] = coor[:, 0] - crop_left
        newcoor[:, 1] = coor[:, 1] - crop_top
        newcoor = newcoor[(newcoor[:, 0] >= 0) & (newcoor[:, 0] < crop_width) & (newcoor[:, 1] >= 0) & (newcoor[:, 1] < crop_height)]
        newcoords.append(newcoor)
    return newcoords

def split_n_crop(image, coords, tile_size):
    image = torch.tensor(image)
    N_tiles = image.shape[-2] // tile_size
    M_tiles = image.shape[-1] // tile_size
    image = v2.functional.crop(image, 0, 0, tile_size*N_tiles, tile_size*M_tiles)
    newimage = image.reshape(-1, 4, N_tiles, tile_size, M_tiles, tile_size).permute(0, 2, 4, 1, 3, 5).reshape(-1, 4, tile_size, tile_size)
    newcoords = []
    for coor in coords:
        for i in range(N_tiles):
            for j in range(M_tiles):
                crop_left, crop_top = j*tile_size, i*tile_size
                newcoor = np.zeros_like(coor)
                newcoor[:, 0] = coor[:, 0] - crop_left
                newcoor[:, 1] = coor[:, 1] - crop_top
                newcoor = newcoor[(newcoor[:, 0] >= 0) & (newcoor[:, 0] < tile_size) & (newcoor[:, 1] >= 0) & (newcoor[:, 1] < tile_size)]
                newcoords.append(newcoor)
    newimage = np.array(newimage)
    return newimage, newcoords

def coords_to_treecount(coords):
    treecount = []
    for coor in coords:
        treecount.append(coor.shape[0])
    return np.array(treecount)