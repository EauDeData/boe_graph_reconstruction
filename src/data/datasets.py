import pytesseract

from src.data.defaults import REPLACE_PATH_EXPR, BASE_JSONS, STOPWORDS, MAX_WORDS

import os
import math
import torch
import json
import copy
import random
from tqdm import tqdm
import numpy as np
import networkx as nx
from PIL import Image

random.seed(42)


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
def split_string_n_chunks(s, n):
    if n <= 0:
        raise ValueError("Number of chunks must be greater than 0")
    chunk_size = len(s) // n
    remainder = len(s) % n

    chunks = []
    start = 0
    for i in range(n):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(s[start:end])
        start = end

    return chunks
class BOEDataset:
    def __init__(self, json_split_txt, base_jsons=BASE_JSONS,
                 replace_path_expression=REPLACE_PATH_EXPR, random_walk_leng=1, transforms = lambda x: x,
                 tokenizer=lambda x: x):
        self.replace = eval(replace_path_expression)
        self.documents = []
        for line in tqdm(open(os.path.join(base_jsons, json_split_txt)).readlines()):
            path = os.path.join(base_jsons, line.strip()).replace('jsons_gt', 'graphs_gt')

            # Avoid empty files
            if len(json.load(open(path))['pages']['0']) > 1:
                self.documents.append(path)

        self.random_walk_leng = random_walk_leng
        self.transforms = transforms

        # at this point this is not even coding this is bullshitting
        # self.transforms.transforms = [resize_image] + self.transforms.transforms[2:]
        self.tokenizer=tokenizer
            
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
    @staticmethod
    def phoc_encode(word, levels = 3):
        word = ''.join(char for char in word if ord(char) < 128)
        descriptor = []
        for L in range(levels):
            splitted_word = split_string_n_chunks(word, L + 1)

            for piece in splitted_word:
                level = [0 for _ in range(128)]
                for char in piece:
                    level[ord(char)] += 1
                descriptor.extend(level)

        return np.array(descriptor)

    def crop_dataset(self):

        for idx in tqdm(list(range(len(self))), desc='Cropping dataset'):
            path = self.documents[idx]

            json_data = json.load(open(path))
            impath = (json_data['path'].replace(*self.replace)
                                     .replace('images', 'numpy').replace('.pdf', '.npz'))

            folder_path = impath.replace('.npz', '_cropped_ocrs').replace('numpy', 'crops')
            os.makedirs(folder_path, exist_ok=True)
            crop_path = os.path.join(folder_path, 'crop.png')

            if not os.path.exists(crop_path):
                page = json_data['topic_gt']["page"]
                segment = json_data['topic_gt']["idx_segment"]
                x, y, x2, y2 = json_data['pages'][page][segment]['bbox']

                image = Image.fromarray(np.load(impath)[page][y:y2, x:x2])
                image.save(crop_path)
            else:
                image = Image.open(crop_path)

            df = pytesseract.image_to_data(image, lang='spa_old', output_type=pytesseract.Output.DATAFRAME)
            df = df[df['level'] == 5].reset_index()

            words_path = os.path.join(folder_path, 'words')
            metadata_path = os.path.join(folder_path, 'metadata')

            os.makedirs(words_path, exist_ok=True)
            os.makedirs(metadata_path, exist_ok=True)

            for i in df.index:

                left = df.at[i, 'left']
                top = df.at[i, 'top']
                width = df.at[i, 'width']
                height = df.at[i, 'height']
                right = left + width
                bottom = top + height
                # Crop the image using the bounding box coordinates
                try:
                    cropped_image = image.crop((left, top, right, bottom))
                    cropped_image.save(os.path.join(words_path, f"{i}.png"))
                    with open(os.path.join(metadata_path, f'{i}.txt'), 'w') as handler:
                        handler.write('\n'.join([f'left: {left}', f'top: {top}',
                                                 f'width: {left}', f'height: {left}',
                                                 f"word: {df.at[i, 'text']}"]))
                except SystemError: continue

    def __getitem__(self, idx):
        path = self.documents[idx]
        json_data = json.load(open(path))
        # TODO: Is image too slow?
        # impath = (json_data['path'].replace(*self.replace)
        #                          .replace('images', 'numpy').replace('.pdf', '.npz'))
        #
        # page = json_data['topic_gt']["page"]
        # segment = json_data['topic_gt']["idx_segment"]
        # x, y, x2, y2 = json_data['pages'][page][segment]['bbox']
        #
        # image = Image.fromarray(np.load(impath)[page][y:y2, x:x2])
        crop_path = ((json_data['path'].replace(*self.replace)
                                 .replace('images', 'numpy')
                      .replace('.pdf', '.npz'))
                     .replace('.npz', '_cropped_ocrs')
                     .replace('numpy', 'crops')
                     )
        words = os.path.join(crop_path, 'words')

        # HGHAHAHAHHHAHAHAHAHA TRY TO READ THIS CODE FUCKER
        wordcrops = [Image.open(os.path.join(crop_path, words, imname)).convert('RGB')
                     for imname in os.listdir(words) if
                     len(open(os.path.join(crop_path, words, imname)
                              .replace('words', 'metadata').replace('.png', '.txt')
                          ).readlines()[-1].strip()) > 3]
        if not len(wordcrops):
            wordcrops = [Image.new('RGB', (5, 5))]
        random.shuffle(wordcrops)
        wordcrops = wordcrops[:MAX_WORDS]

        ocr = json_data['ocr_gt']
        query = json_data['query']
        # date = json_data['date'].split('/')[-1]
        #
        # splitted_ocr = ocr.split(' ')
        # phoc_ocr = torch.tensor([self.phoc_encode(ocr).tolist() for ocr in splitted_ocr], dtype=torch.float32)
        #
        # splitted_query = query.split(' ')
        # phoc_query = torch.tensor([self.phoc_encode(ocr).tolist() for ocr in splitted_query], dtype=torch.float32)
        #
        # phoc_dates = torch.tensor(self.phoc_encode(date), dtype=torch.float32)

        return {
            # 'image': self.transforms(image),
            'ocr': ocr,
            'query': [self.tokenizer(q) for q in query.split(' ') if not q in STOPWORDS],
            'images': wordcrops
            # 'date': date,
            # 'phoc_ocr': phoc_ocr,
            # 'phoc_query': phoc_query,
            # 'phoc_dates': phoc_dates
        }