import pytesseract

from src.data.defaults import REPLACE_PATH_EXPR, BASE_JSONS, STOPWORDS, MAX_WORDS
import PIL.ImageOps as imOps
import cv2
import unicodedata
from src.data.datautils import extract_center
import os
import math
import torch
import json
import shutil

import copy
import random
from tqdm import tqdm
import numpy as np
import networkx as nx
from PIL import Image,  ImageDraw, ImageFont
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.stem import SnowballStemmer
from src.data.defaults import IMAGE_SIZE
# Make sure to download stopwords and punkt if you haven't already
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
random.seed(42)

SPANISH_STOPWORDS = set(stopwords.words('spanish'))

def sanitize_string(input_string: str) -> str:
    # Normalize the input string to its decomposed form (NFD), separating characters from their accents
    normalized_string = unicodedata.normalize('NFD', input_string)

    # Filter out all diacritical marks (i.e., non-spacing marks like accents)
    sanitized_string = ''.join(char for char in normalized_string if not unicodedata.combining(char))

    return sanitized_string

def create_template_image(word, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2,
                             color=(255, 255, 255)):

    word = sanitize_string(word)

    # Create a blank image
    image_size = (500, 500)  # You can adjust the size as needed
    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    # Set the text size and position
    text_size, _ = cv2.getTextSize(word, font, font_scale, font_thickness)
    text_width, text_height = text_size
    text_x = (image_size[0] - text_width) // 2
    text_y = (image_size[1] + text_height) // 2

    # Put the text on the image
    cv2.putText(image, word, (text_x, text_y), font, font_scale, color, font_thickness)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours in the image
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box of the first contour (assuming there's only one word)
    all_contours = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_contours)
    cropped_image = image[y:y + h, x:x + w]
    return Image.fromarray(255 - cropped_image).resize(IMAGE_SIZE)

def split_by_punctuation_and_spaces(sentence):
    # Split by any punctuation or spaces using regex
    tokens = re.split(r'\W+', sentence)

    # Remove empty tokens
    tokens = [token for token in tokens if token if not token in SPANISH_STOPWORDS]

    return tokens

def curate_token(token, stemmer):
    return stemmer(token)

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
            # if len(self.documents) > 65: break
            # Avoid empty files
            if len(json.load(open(path))['pages']['0']) > 1:
                self.documents.append(path)

        self.random_walk_leng = random_walk_leng
        self.transforms = transforms
        self.stemmer = SnowballStemmer('spanish').stem

        # at this point this is not even coding this is bullshitting
        # self.transforms.transforms = [resize_image] + self.transforms.transforms[2:]
        # TODO: DONT HARDCODE THIS TOKENIZER LOL
        tokepath = './tokenizer.json'
        if os.path.exists(tokepath):
            self.tokenizer = json.load(open(tokepath))
        else:
            assert not 'test' in json_split_txt, 'tokenizer might not be computed on test partition!!'
            self.tokenizer = self.create_token_dict()
            json.dump(self.tokenizer, open(tokepath, 'w'), indent=2)
        # self.tokenizer=json.load(open('/data2/users/amolina/oda_ocr_output/oda_giga_tokenizer.json'))
            
    def __len__(self):
        return len(self.documents)

    def create_token_dict(self):
        token_dict = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2}
        # Get Spanish stopwords
        # Token ID starting point (after special tokens)

        # Tokenize and process each sentence
        for doc_path in tqdm(self.documents):
            json_data = json.load(open(doc_path))
            tokens = split_by_punctuation_and_spaces(json_data['query'].lower())  # Tokenize the sentence
            # tokens = tokens + split_by_punctuation_and_spaces(json_data['ocr_gt'].lower())
            for token in tokens:
                token = curate_token(token, self.stemmer)
                if not token in token_dict:
                    token_dict[token] = len(token_dict)

        return token_dict

    def tokenize_sentence(self, sentence):
        tokens = [curate_token(t.lower(), self.stemmer) for t in sentence]  # Tokenize the sentence
        return [self.tokenizer['[CLS]']] + [self.tokenizer.get(token, self.tokenizer['[UNK]']) for token in tokens]

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

        crop_path = ((json_data['path'].replace(*self.replace)
                                 .replace('images', 'numpy')
                      .replace('.pdf', '.npz'))
                     .replace('.npz', '_cropped_ocrs')
                     .replace('numpy', 'crops')
                     )
        words = os.path.join(crop_path, 'words')

        # HGHAHAHAHHHAHAHAHAHA TRY TO READ THIS CODE FUCKER
        croplist = [os.path.join(crop_path, words, x) for x in sorted(os.listdir(words), key=lambda x: int(x.split('.')[0]))]
        metadatas = [imname.replace('words', 'metadata').replace('.png', '.txt') for imname in croplist]
        # idxs = list(range(len(croplist)))
        # random.shuffle(idxs)
        # idxs = idxs[:MAX_WORDS]

        # croplist = [croplist[i] for i in idxs]
        # metadatas = [metadatas[i] for i in idxs]

        assert  len(croplist) == len(metadatas), f"CROPS: {croplist}\nMETADATAS: {metadatas}"
        MIN_SIZE = 2

        metadata_files = [open(metadata).readlines() for metadata in metadatas]
        # centers = [extract_center(lines) for lines in metadata_files
        #            if len(lines[-1].strip().split(': ')[-1]) > MIN_SIZE]

        # Extract the words from metadata files and filter based on MIN_SIZE
        words_thing = [file[-1].strip().split(': ')[-1] for file in metadata_files
                       if len(file[-1].strip().split(': ')[-1]) > MIN_SIZE][:MAX_WORDS]
        # crop_image = Image.open(os.path.join(crop_path, 'crop.png'))
        # crop_image.save('tmp/crop.png')

        # shutil.copytree(crop_path, 'tmp/crops', dirs_exist_ok=True)

        # Assuming 'croplist' corresponds to images, and 'words_thing' has matching entries
        wordcrops = [imOps.invert(Image.open(imname).convert('RGB').resize(IMAGE_SIZE))
                     for imname, metadata in zip(croplist, metadata_files)
                     if len(metadata[-1].strip().split(': ')[-1]) > MIN_SIZE][:MAX_WORDS]


        if not len(wordcrops):
            wordcrops = [Image.new('RGB', (5, 5))]
            words_thing = []

        ocr = json_data['ocr_gt']
        query = json_data['query']
        query_str =  [q for q in split_by_punctuation_and_spaces(query)][:MAX_WORDS]
        queries_pils = [imOps.invert(create_template_image(t) ) for t in query_str]
        # print(query_str, len(query_str))
        # print('pils of query:', len(queries_pils))
        # print('tokenized len: ', len( self.tokenize_sentence(query_str)))
        # print( self.tokenize_sentence(query_str))

        return {
            'ocr': ocr,
            'query': self.tokenize_sentence(query_str),
            'images': wordcrops,
            'words': words_thing,
            # TODO: Also we need to compute visual prototypes of each token
            'query_str': query_str,
            'query_img': queries_pils

        }