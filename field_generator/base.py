import torch

# Base kernels

def normalize_positive_tensor(tensor):
    max, min = torch.max(tensor), torch.min(tensor)
    return (tensor - min) / (max - min)

def normalize_tensor(tensor):
    max, min = torch.max(tensor), torch.min(tensor)
    return tensor / torch.maximum(max.abs(), min.abs())

def gaussian_kernel(kernel_size: int, sigma: float, device="cpu"):
    """Generates a 2D Gaussian kernel."""
    eps = 1e-6
    x = torch.arange(kernel_size, device=device).float() - (kernel_size - 1) / 2
    gauss = torch.exp(-x**2 / (2 * (sigma+eps)**2))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return (kernel_2d / kernel_2d.sum()).unsqueeze(0).unsqueeze(0)

def gaussian_filter(input, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma, device=input.device)
    groups = 1
    if input.ndim == 3:
        if input.shape[0] > 1:
            kernel = kernel.repeat([input.shape[0], 1, 1, 1])
            groups = input.shape[0]
    if input.ndim == 4:
        if input.shape[1] > 1:
            kernel = kernel.repeat([input.shape[1], 1, 1, 1])
            groups = input.shape[1]
    tmp = torch.nn.functional.conv2d(input, kernel, padding="same", groups=groups)
    return tmp