# Field Generator

A tool for generating synthetic field images with trees for computer vision applications. This project creates realistic aerial/satellite-like images of agricultural fields with various tree distributions, shadows, and environmental features.

## Features

- **Realistic Field Generation**: Creates synthetic field images with configurable tree distributions
- **Advanced Rendering**: Includes shadows, lighting effects, and various background textures
- **Batch Processing**: Generate multiple fields efficiently with GPU acceleration
- **Configurable Parameters**: Extensive configuration options for all aspects of generation
- **CUDA Support**: Optimized for GPU acceleration with PyTorch
- **Docker Ready**: Containerized for easy deployment and reproducibility

## Installation

### Using Docker (Recommended)

```bash
# Build the Docker image
docker build -t field-generator .

docker run --gpus all -it --rm -v "$(pwd)/outputs":/app/outputs field-generator bash
```

### Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the generator
python main.py
```

## Usage

### Basic Usage

```bash
# Generate default field
python main.py

# Generate with custom parameters
python main.py --total_fields 5 --field_size 640 640 --device cuda:0
```

### Configuration

The generator supports extensive configuration through command-line arguments or by modifying the `DEFAULT_CONFIG` in `src/constants.py`. Key parameters include:

- **Field Properties**: Size, zoom, batch processing
- **Tree Distribution**: Density, positioning, size variations
- **Visual Effects**: Shadows, lighting, color schemes
- **Background**: Patches, overlays, noise patterns
- **Postprocessing**: Tiling, rotation, final adjustments

### Example Configuration

The `DEFAULT_CONFIG` in `src/constants.py` provides the default settings. You can modify it directly or override it via command-line arguments. Example:

```python
config = {
    "field_size": (1000, 1000),
    "batch_size": 10,
    "total_fields": 50,
    "tree_threshold": 0.3,
    "shadow_length": 15,
    "output_dir": "outputs",
    "verbose": True,
    # ...additional options
}
```

## Output

The generator creates:
- **output.npy**: Generated field images
- **coordinates.npz**: Tree coordinate data
- **count.npy**: Tree count information
- **config.ini**: Configuration used for generation

## Requirements

- Python 3.10+
- PyTorch 2.3+
- NVIDIA GPU (recommended)
- Docker (for containerized deployment)

## License

This project is licensed under the MIT License.
