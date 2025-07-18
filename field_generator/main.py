"""Main entry point for field generator."""

import os

import numpy as np

from src.constants import DEFAULT_CONFIG
from src.field_generator import generate_field
from src.postprocessing import postprocessing
from src.argmanager import get_parser_from_dict, save_config_to_ini, update_config_from_args


def main() -> None:
    """Main function to generate field images."""
    # Parse command-line arguments
    parser = get_parser_from_dict(DEFAULT_CONFIG)
    args = parser.parse_args()

    # Update config from args
    config = DEFAULT_CONFIG.copy()
    config = update_config_from_args(config, args)

    # Ensure output directory exists
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
        

    # Save final config to .ini file
    save_config_to_ini(config, os.path.join(config["output_dir"], "config.ini"))

    # Print configuration
    if config["verbose"]:
        print("Configuration:")
        for k, v in config.items():
            print(f"{k}: {v}")

    # Generate fields
    outputs_list = []
    coordinates_list = []
    count_list = []

    for _ in range(config["total_fields"] // config["batch_size"]):
        outputs, coordinates, count = generate_field(**config)
        outputs_list.append(outputs)
        coordinates_list += coordinates
        count_list += count

    outputs = np.concatenate(outputs_list)
    coordinates = coordinates_list
    count = count_list

    # Apply postprocessing
    outputs, coordinates, count = postprocessing(outputs, coordinates, **config)

    # Save outputs and labels
    output_path = os.path.join(config["output_dir"], "output.npy")
    coordinates_path = os.path.join(config["output_dir"], "coordinates.npz")
    count_path = os.path.join(config["output_dir"], "count.npy")

    np.save(output_path, outputs)
    np.savez(coordinates_path, **{f"coords{i}": t for i, t in enumerate(coordinates)})
    np.save(count_path, count)


if __name__ == "__main__":
    main()
