import numpy as np
import networkx as nx


def extract_center(metadata_lines):
    # Create a dictionary to store metadata values
    metadata_dict = {}

    for line in metadata_lines:
        try:
            key, value = line.split(':')

            metadata_dict[key.strip()] = int(value.strip())
        except ValueError:
            pass # The key is not useful if can't be converted to int

    # Calculate the center
    x_center = metadata_dict['left'] + metadata_dict['width'] / 2
    y_center = metadata_dict['top'] + metadata_dict['height'] / 2

    return (x_center, y_center)

