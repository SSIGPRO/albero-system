import torch
import random
from torchvision.transforms import v2
from base import gaussian_filter, normalize_positive_tensor, normalize_tensor

def generate_probability_maps(prob_map_size,
                              batch_size=1,
                              filter_size=51,
                              filter_sigma=6,
                              noise_level=0.01,
                              sparse_remove_rate=0.05,
                              sparse_add_rate=0.05,
                              row_prune_rate=0.01,
                              column_prune_rate=0.01,
                              final_filter_size=21,
                              final_filter_sigma=0.5,
                              device="cpu"):
        # generate clustered probability map
        prob_map = torch.randn((batch_size, *prob_map_size), device=device)
        prob_map = gaussian_filter(prob_map, filter_size, filter_sigma)

        # normalize between -1 and 1
        max = torch.max(prob_map)
        prob_map = prob_map / max

        # randomly prune columns
        sparse_map = torch.rand((prob_map.shape[0], prob_map.shape[-2]), device=device)
        remove_mask = sparse_map < column_prune_rate
        prob_map[remove_mask] = torch.min(prob_map)

        # randomly prune rows
        sparse_map = torch.rand((prob_map.shape[0], prob_map.shape[-1]), device=device)
        remove_mask = sparse_map < row_prune_rate
        prob_map = prob_map.permute(0, 2, 1)  # transpose to prune rows
        prob_map[remove_mask] = torch.min(prob_map)
        prob_map = prob_map.permute(0, 2, 1)  # transpose back to original shape

        # randomly prune/add sparse values
        sparse_map = torch.rand_like(prob_map)
        add_mask = sparse_map > 1-sparse_add_rate
        remove_mask = sparse_map < sparse_remove_rate
        prob_map[remove_mask] = torch.min(prob_map)
        prob_map[add_mask] = torch.max(prob_map)

        # filter again map
        prob_map = gaussian_filter(prob_map, final_filter_size, final_filter_sigma)

        # generate and add noise to map
        noise = torch.randn((batch_size, *prob_map_size), device=device) * noise_level
        prob_map = prob_map*(1 + noise)

        # normalize between 0 and 1
        prob_map = normalize_positive_tensor(prob_map)

        return prob_map

def prob_to_treesize(prob_map, threshold, treeshape_min_size, treeshape_size, treeshape_max_size, steepness=1.0, distribution_shift=0.0):
    tree_boolmap = prob_map > threshold
    treesize_map = torch.nn.functional.sigmoid((prob_map-threshold)/(1-threshold)*steepness+distribution_shift)*treeshape_size
    treesize_map = torch.clamp(treesize_map, treeshape_min_size, treeshape_max_size)
    treesize_map = treesize_map*tree_boolmap.to(torch.float32)
    return treesize_map, tree_boolmap

def generate_tree_offsets(treesize_map,
                          center_jitter=(0, 0),
                          alternate_offset=0):
    device = treesize_map.device
    # Generate random offsets
    if(center_jitter[0] == 0 and center_jitter[1] == 0):
        # No jitter
        tree_offsets = torch.stack((torch.zeros_like(treesize_map, device=device),
                                    torch.zeros_like(treesize_map, device=device)), dim=1)
    else:
        tree_offsets = torch.stack((torch.randint_like(treesize_map, low=-center_jitter[0], high=center_jitter[0]),
                                    torch.randint_like(treesize_map, low=-center_jitter[1], high=center_jitter[1])), dim=1)
    # Add alternate offset to odd rows
    if alternate_offset > 0:
        tree_offsets[:, 0, 1::2] += alternate_offset

    return tree_offsets

# Draws a filled noisy circle on a tensor using vectorized operations.
def generate_tree_sprites(treesize_map,
                          tree_sprite_size,
                          max_tree_size,
                          tree_offsets,
                          noise_level=3.0,
                          filter_size=5,
                          filter_sigma=2.0):
    device = treesize_map.device
    batch_size = treesize_map.shape[0]
    map_N, map_M = treesize_map.shape[1], treesize_map.shape[2]
    # Sprite size
    N = tree_sprite_size*2    

    # Center of the circle
    center = (N//2, N//2)
    y_coords, x_coords = torch.meshgrid(torch.arange(N, device=device), torch.arange(N, device=device), indexing='ij')
    
    # Compute distances from the center+offset for all pixels
    x_coords = x_coords.unsqueeze(0).unsqueeze(0).expand(batch_size, map_N, map_M, -1, -1)
    y_coords = y_coords.unsqueeze(0).unsqueeze(0).expand(batch_size, map_N, map_M, -1, -1)
    
    tree_offsets = tree_offsets.permute(1, 0, 2, 3) # permute to (2, batch_size, N, N)
    x_offsets = tree_offsets[0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, N, N)
    y_offsets = tree_offsets[1].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, N, N)
    dist = torch.sqrt((x_coords - center[0] - x_offsets) ** 2 + (y_coords - center[1] - y_offsets) ** 2)

    del x_coords, y_coords, x_offsets, y_offsets
    
    # Generate noisy radius values
    noise = (torch.rand(*treesize_map.shape, N, N, device=device) - 0.5) * 2 * noise_level # uniform noise between -noise_level and +noise_level
    noise = gaussian_filter(noise.view(-1, noise.shape[3], noise.shape[3]), filter_size, filter_sigma).view(*noise.shape)
    radius = treesize_map
    noisy_radius = torch.clamp(radius.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, N, N)+noise, 0, max_tree_size)

    del noise
    
    # Fill in the circle (set to color where distance <= noisy radius)
    tree_sprites = (dist <= noisy_radius)*torch.sqrt(torch.clamp(noisy_radius**2-dist**2, 0, None)/(torch.max(noisy_radius)**2)) # circle-like shading
    
    return tree_sprites

def _compose_stack(sprites, portion_num, overlap=(0, 0)):
    batch_size = sprites.shape[0]
    i_portion = portion_num//5
    j_portion = portion_num%5
    # pad sprites on the first 2 dimensions (add empty trees) to make them divisible by 5
    pad_i = 0 if sprites.shape[1]%5 == 0 else (5-sprites.shape[1]%5)
    pad_j = 0 if sprites.shape[2]%5 == 0 else (5-sprites.shape[2]%5)
    sprites = torch.nn.functional.pad(sprites.permute(3, 4, 0, 1, 2), (0, pad_j, 0, pad_i), mode="constant").permute(2, 3, 4, 0, 1)
    # single sprites area
    N, M = sprites.shape[-2:]
    N -= overlap[1]
    M -= overlap[0]
    # shift dependent on portion
    left_pad, right_pad, up_pad, down_pad = 0, 4*M-overlap[0], 0, 4*N-overlap[1]
    left_overall_pad, right_overall_pad, up_overall_pad, down_overall_pad = 0, 0, 0, 0
    if i_portion == 0:
        up_overall_pad, down_overall_pad = 0, 4*N
    if i_portion == 1:
        up_overall_pad, down_overall_pad = N, 3*N
    if i_portion == 2:
        up_overall_pad, down_overall_pad = 2*N, 2*N
    if i_portion == 3:
        up_overall_pad, down_overall_pad = 3*N, N
    if i_portion == 4:
        up_overall_pad, down_overall_pad = 4*N, 0
    if j_portion == 0:
        left_overall_pad, right_overall_pad = 0, 4*M
    if j_portion == 1:
        left_overall_pad, right_overall_pad = M, 3*M
    if j_portion == 2:
        left_overall_pad, right_overall_pad = 2*M, 2*M
    if j_portion == 3:
        left_overall_pad, right_overall_pad = 3*M, M
    if j_portion == 4:
        left_overall_pad, right_overall_pad = 4*M, 0
    # pad sprites to add spacing between trees
    sprites = torch.nn.functional.pad(sprites, (left_pad, right_pad, up_pad, down_pad), mode="constant")
    N, M = sprites.shape[-2:]
    batch_size = sprites.shape[0]
    res = sprites[:, i_portion::5, j_portion::5]
    res = res.reshape(batch_size, sprites.shape[1]//5, sprites.shape[2]//5, N, M)
    res = res.permute(0, 1, 3, 2, 4)
    res = res.reshape(batch_size, sprites.shape[1]//5*N, sprites.shape[2]//5*M)
    return torch.nn.functional.pad(res, (left_overall_pad, right_overall_pad, up_overall_pad, down_overall_pad), mode="constant")

def _stack_to_map(sprites, overlap=(0, 0)):
    fieldmap = _compose_stack(sprites, 0, overlap)
    for i in range(1, 5*5):
        fieldmap = torch.maximum(fieldmap, _compose_stack(sprites, i, overlap))
    return fieldmap

def generate_field_mask(tree_sprites, field_size, distance, offset):
    distance = int(round(distance[0])), int(round(distance[1]))
    N, M = tree_sprites.shape[-2:]
    overlap = (N-distance[0], M-distance[1])
    # Put together tree sprites
    field_mask = _stack_to_map(tree_sprites, overlap=overlap)
    field_mask = torch.nn.functional.pad(field_mask, (offset, 0, offset, 0), mode="constant")
    # Cut/expand the canvas to the size of the field
    result = torch.zeros((field_mask.shape[0], *field_size), device=tree_sprites.device)
    result[:, :min(field_size[0], field_mask.shape[1]), :min(field_size[1], field_mask.shape[2])] = field_mask[:, :field_size[0], :field_size[1]]
    return result

def _shift_towards_direction(tensor, delta, theta):
    # Get the unit vector for the direction
    theta_tensor = torch.tensor(theta, device=tensor.device)
    dx = torch.cos(theta_tensor)
    dy = torch.sin(theta_tensor)

    # Calculate integer steps for indexing
    step_x = torch.round(dx*delta).int()
    step_y = torch.round(dy*delta).int()
    
    delta = int(round(delta))

    # Pad the tensor to handle edge cases
    if tensor.ndim > 2:
        padded_tensor = torch.nn.functional.pad(tensor, (delta, delta, delta, delta), mode='constant')
        # Shift the tensor in the direction of the unit vector
        shifted_tensor = padded_tensor[...,
                                       delta + step_y : delta + step_y + tensor.shape[-2],
                                       delta + step_x : delta + step_x + tensor.shape[-1]]
    else:
        padded_tensor = torch.nn.functional.pad(tensor.unsqueeze(0), (delta, delta, delta, delta), mode='constant').squeeze()
        # Shift the tensor in the direction of the unit vector
        shifted_tensor = padded_tensor[delta + step_y : delta + step_y + tensor.shape[-2],
                                       delta + step_x : delta + step_x + tensor.shape[-1]]
    
    return shifted_tensor

def project_shadow(tensor, iterations, strength, rad_dir, filter_size=21, filter_sigma=1):
    projection_tensor = torch.zeros_like(tensor)
    shifted_tensor = torch.empty_like(tensor)
    shifted_tensor[...] = tensor[...]

    for _ in range(iterations):
        shifted_tensor = _shift_towards_direction(shifted_tensor, strength/iterations, rad_dir)
        projection_tensor = torch.maximum(projection_tensor, shifted_tensor)

    projection_tensor = gaussian_filter(projection_tensor, filter_size, filter_sigma).squeeze()
    projection_tensor = normalize_positive_tensor(projection_tensor)

    return projection_tensor

def compute_gradient_map(tensor, theta):
    shifted_tensor = _shift_towards_direction(tensor, 1, theta)

    # Compute the differences
    differences = (tensor-shifted_tensor)

    return normalize_tensor(differences)

def apply_lighting(field, field_gradient, strength, filter_size=7, filter_sigma=3.0):

    lighting_mask = gaussian_filter(field_gradient, filter_size, filter_sigma) # smooth the gradient map
    lighting_mask = normalize_tensor(lighting_mask) # normalize again the gradient map

    for i in range(field.shape[0]):
        field[i] = field[i]*(1+lighting_mask*strength)

    return field

def bool_tensor_to_coords(bool_tensor, offsets, constant_offset, dx, dy) -> torch.Tensor:
    device = bool_tensor.device
    _, H, W = bool_tensor.shape
    y_coords = torch.arange(H, device=device).view(H, 1).expand(H, W)*dy+offsets[:, 1]+constant_offset[1]
    x_coords = torch.arange(W, device=device).view(1, W).expand(H, W)*dx+offsets[:, 0]+constant_offset[0]
    coords_map = torch.stack((x_coords, y_coords), dim=1)  # (B, H, W, 2)

    return coords_map

def coords_map_to_list(coords_map, bool_tensor, sizes):
    B = bool_tensor.shape[0]
    coords_list = []
    for b in range(B):
        y = coords_map[b, 1][bool_tensor[b]].flatten()
        x = coords_map[b, 0][bool_tensor[b]].flatten()
        s = sizes[b][bool_tensor[b]].flatten()
        coords = torch.stack((x, y, s), dim=1)  # (N, 3)
        coords_list.append(coords)
    return coords_list

def generate_rectangle_coords(batch_size, max_width, max_height, canvas_size, device="cpu"):
    # Sample top-left corner coordinates
    xleft = torch.randint(0, canvas_size[0] - max_width, (batch_size,), device=device)
    ytop = torch.randint(0, canvas_size[1] - max_height, (batch_size,), device=device)

    # Sample widths and heights that satisfy the max constraints
    widths = torch.randint(1, max_width + 1, (batch_size,), device=device)
    heights = torch.randint(1, max_height + 1, (batch_size,), device=device)

    # Compute bottom-right coordinates
    xright = xleft + widths
    ybottom = ytop + heights

    # Stack to get final (batch_size, 4) tensor: (xleft, ytop, xright, ybottom)
    patches_coords = torch.stack([xleft, ytop, xright, ybottom], dim=1)

    return patches_coords

def soft_rectangle_mask(H, W, coords, sharpness=0.2, device="cpu"):
    """
    Generate N soft masks of shape (H, W), where each mask contains a soft-edged rectangle.

    Args:
        H, W: spatial dimensions of the canvas
        coords: tensor of shape (N, 4) with (x1, y1, x2, y2) per rectangle
        sharpness: how steep the sigmoid edge is

    Returns:
        mask: (N, H, W) soft rectangle masks
    """
    coords = coords.to(device)
    N = coords.shape[0]

    x1, y1, x2, y2 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    x_min = torch.min(x1, x2).to(device)
    x_max = torch.max(x1, x2).to(device)
    y_min = torch.min(y1, y2).to(device)
    y_max = torch.max(y1, y2).to(device)

    # Coordinate grids
    ys = torch.linspace(0, H - 1, H, device=device).view(1, H, 1)  # (1, H, 1)
    xs = torch.linspace(0, W - 1, W, device=device).view(1, 1, W)  # (1, 1, W)

    # Reshape bounds for broadcasting
    x_min = x_min.view(N, 1, 1)
    x_max = x_max.view(N, 1, 1)
    y_min = y_min.view(N, 1, 1)
    y_max = y_max.view(N, 1, 1)

    # Compute sigmoid transitions (shape: N × H × W)
    left   = torch.sigmoid(sharpness * (xs - x_min))   # (N, H, W)
    right  = torch.sigmoid(sharpness * (x_max - xs))   # (N, H, W)
    top    = torch.sigmoid(sharpness * (ys - y_min))   # (N, H, W)
    bottom = torch.sigmoid(sharpness * (y_max - ys))   # (N, H, W)

    mask = left * right * top * bottom  # (N, H, W)

    return mask