from src.data.defaults import REPLACE_PATH_EXPR, BASE_JSONS

import os
import math
import json
import copy
import random
from tqdm import tqdm
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
                 replace_path_expression=REPLACE_PATH_EXPR, random_walk_leng=2):
        self.replace = eval(replace_path_expression)
        self.documents = []
        for line in tqdm(open(os.path.join(base_jsons, json_split_txt)).readlines()):
            path = os.path.join(base_jsons, line.strip()).replace('jsons_gt', 'graphs_gt')

            # Avoid empty files
            if len(json.load(open(path))['pages']['0']):
                self.documents.append(path)

        self.random_walk_leng = random_walk_leng
            
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

            graph.nodes[node_in]['centroid_scaled'] = (x_ini, y_ini)
            graph.nodes[node_out]['centroid_scaled'] = (x_fin, y_fin)


            distance = math.sqrt((x_ini - x_fin) ** 2 + (y_ini - y_fin) ** 2)

            graph[node_in][node_out]['gt'] = distance


        return graph, images, gt
    @staticmethod
    def random_walk(graph, leng, seeds):

        nodes = list(graph.nodes())
        if not len(nodes): return [seeds]

        random_walks = []
        for current_node in seeds:
            this_walk = [current_node]
            for _ in range(leng):
                possible_paths = list(graph.neighbors(current_node))
                if not len(possible_paths): break
                this_walk.append(random.choice(possible_paths))
                current_node = this_walk[-1]
            random_walks.append(this_walk)
        return set(sum(random_walks, start=[]))
    def __getitem__(self, idx):
        path = self.documents[idx]
        json_data = json.load(open(path))

        impath = (json_data['path'].replace(*self.replace)
                  .replace('images', 'numpy').replace('.pdf', '.npz'))
        graph_path = impath.replace('.npz', '.gml')

        graph, images, gt = self.parse_graph_data(nx.read_gml(graph_path), np.load(impath)['0'])

        target_ocr = max(json_data['pages']['0'], key=lambda x: x['similarity'] if 'similarity' in x else 0)

        selected_initial = [x for x in graph.nodes() if graph.nodes[x]['ocr'] == target_ocr['ocr']][0]
        nodes_walk = nx.ego_graph(graph, selected_initial) # self.random_walk(graph, self.random_walk_leng, seeds=[selected_initial]*2)

        if len(nodes_walk) > 1:
            graph = nx.subgraph(graph, [x for x in nodes_walk if selected_initial!=x])

        return {
            'graph': graph,
            'input_data': images,
            'gt': gt,
            'query': json_data['query']
        }

