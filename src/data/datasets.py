from src.data.defaults import REPLACE_PATH_EXPR, BASE_JSONS

import os
import math
import json
import copy

import numpy as np
import networkx as nx
from PIL import Image

def calculate_angle(x, y, x1, y1):
    # Calculate dot product
    dot_product = x * x1 + y * y1

    # Calculate magnitudes
    magnitude_1 = math.sqrt(x ** 2 + y ** 2)
    magnitude_2 = math.sqrt(x1 ** 2 + y1 ** 2)

    # Calculate cosine of the angle
    cosine_angle = min((dot_product / (magnitude_1 * magnitude_2 + 0.0001)), 1)
    # Calculate the angle in radians
    angle_radians = math.acos(cosine_angle)

    # Convert radians to degrees
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees / 90

class BOEDataset:
    def __init__(self, json_split_txt, base_jsons=BASE_JSONS,
                 replace_path_expression=REPLACE_PATH_EXPR):
        self.replace = eval(replace_path_expression)
        self.documents = []
        for line in open(os.path.join(base_jsons, json_split_txt)).readlines():
            path = os.path.join(base_jsons, line.strip()).replace('jsons_gt', 'graphs_gt')
            self.documents.append(path)
            
    def __len__(self):
        return len(self.documents)

    @staticmethod
    def parse_graph_data(graph: nx.Graph, image: np.ndarray):
        # Here we transform the graph into something we can actually use
        # Assign "is_edge=False" to all nodes
        image_height, image_width, _ = image.shape
        # image_height /= 2
        # image_width /= 2

        images = {}
        gt = {}
        for node in graph.nodes():
            x1, y1, x2, y2 = graph.nodes[node]['bbox']
            graph.nodes[node]['is_edge'] = False
            graph.nodes[node]['width'] = (x2 - x1) / image_width
            graph.nodes[node]['height'] = (y2 - y1) / image_height
            graph.nodes[node]['aspect_ratio'] = 0 if not (y2 - y1) else (x2 - x1) / (y2 - y1)

            images[node] = {'image': Image.fromarray(image[y1:y2, x1:x2]).resize((224, 224)), 'ocr': graph.nodes[node]['ocr']}

        done = []
        for edge in list(graph.edges()):

            node_in, node_out = edge # From here we want to extract the GT node

            x_ini, y_ini = graph.nodes[node_in]['centroid']
            x_fin, y_fin = graph.nodes[node_out]['centroid']

            # TODO: Check the normalization is not crazy AHH norm
            y_ini /= image_height
            y_fin /= image_height

            x_ini /= image_width
            x_fin /= image_width

            distance = math.sqrt((x_ini - x_fin) ** 2 + (y_ini - y_fin) ** 2)

            # IDK how to operate with the angle
            angle = calculate_angle(x_ini - x_fin, y_ini - y_fin, x_fin, y_fin)

            graph[node_in][node_out]['gt'] = (angle,  distance)

            # done.append(f"{node_in}_{node_out}")
            # node_name = f"pred_{node_in}_{node_out}"
            # graph.add_node(node_name, is_edge=True, angle=angle_degrees, distance=distance)

            # graph.add_edge(node_in, node_name)
            # graph.add_edge(node_out, node_name)

            # gt[node_name] = (angle_degrees, distance)

        return graph, images, gt

    def __getitem__(self, idx):
        path = self.documents[idx]
        json_data = json.load(open(path))
        impath = (json_data['path'].replace(*self.replace)
                  .replace('images', 'numpy').replace('.pdf', '.npz'))
        graph_path = impath.replace('.npz', '.gml')
        graph, images, gt = self.parse_graph_data(nx.read_gml(graph_path), np.load(impath)['0'])
        return {
            'graph': graph,
            'input_data': images,
            'gt': gt
        }

