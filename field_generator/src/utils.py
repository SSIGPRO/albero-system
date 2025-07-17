import torch
from torchvision.transforms import v2


def normalize_positive_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] range."""
    max_val, min_val = torch.max(tensor), torch.min(tensor)
    return (tensor - min_val) / (max_val - min_val)


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor by maximum absolute value."""
    max_val, min_val = torch.max(tensor), torch.min(tensor)
    return tensor / torch.maximum(max_val.abs(), min_val.abs())


def gaussian_kernel(kernel_size: int, sigma: float, device: str = "cpu") -> torch.Tensor:
    """Generate a 2D Gaussian kernel.

    Args:
        kernel_size: Size of the kernel (must be odd)
        sigma: Standard deviation of the Gaussian
        device: Device to create the tensor on

    Returns:
        2D Gaussian kernel tensor with shape (1, 1, kernel_size, kernel_size)
    """
    eps = 1e-6
    x = torch.arange(kernel_size, device=device).float() - (kernel_size - 1) / 2
    gauss = torch.exp(-(x**2) / (2 * (sigma + eps) ** 2))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return (kernel_2d / kernel_2d.sum()).unsqueeze(0).unsqueeze(0)


def gaussian_filter(input_tensor: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Apply Gaussian filter to input tensor.

    Args:
        input_tensor: Input tensor to filter
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian

    Returns:
        Filtered tensor with same shape as input
    """
    kernel = gaussian_kernel(kernel_size, sigma, device=input_tensor.device)
    groups = 1

    if input_tensor.ndim == 3:
        if input_tensor.shape[0] > 1:
            kernel = kernel.repeat([input_tensor.shape[0], 1, 1, 1])
            groups = input_tensor.shape[0]
    elif input_tensor.ndim == 4:
        if input_tensor.shape[1] > 1:
            kernel = kernel.repeat([input_tensor.shape[1], 1, 1, 1])
            groups = input_tensor.shape[1]

    # Pad input to avoid border effects
    pad = kernel.shape[-1] // 2
    tmp = torch.nn.functional.pad(input_tensor, (pad, pad, pad, pad), mode="replicate")

    # Apply the kernel
    tmp = torch.nn.functional.conv2d(tmp, kernel, padding="same", groups=groups)

    # Crop the output to the original size
    tmp = v2.functional.crop(tmp, pad, pad, input_tensor.shape[-2], input_tensor.shape[-1])
    return tmp
