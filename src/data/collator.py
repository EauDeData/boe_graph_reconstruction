import networkx as nx
import numpy as np
import torch
from torchvision import transforms
from src.data.defaults import IMAGE_SIZE

from PIL import  Image
from torch.fx.experimental.unification.unification_tools import keymap


def resize_and_pad_images(images, fixed_height, patch_size, max_allowed_width = 128):
    resized_images = []
    max_width = 0

    # Resize images to fixed height and width multiple of patch_size
    for img in images:
        # aspect_ratio = img.width / img.height
        width, height = img.size

        # new_width = int(aspect_ratio * fixed_height)
        # Adjust new_width to be a multiple of patch_size
        new_width = (width + (patch_size - 1)) // patch_size * patch_size
        # print(new_width)
        new_width = min(new_width, max_allowed_width)
        resized_img = img.resize((new_width, fixed_height))
        resized_images.append(resized_img)

        if new_width > max_width:
            max_width = new_width

    # Add padding to make all images the same width
    padded_images = []
    for img in resized_images:
        pad_width = max_width - img.width
        # Create a new image with padding
        new_img = Image.new('RGB', (max_width, fixed_height))
        new_img.paste(img, (0, 0))

        # Adding patch_size x fixed_height blocks of padding if necessary
        if pad_width > 0:
            padding_block = Image.new('RGB', (pad_width, fixed_height))
            new_img.paste(padding_block, (img.width, 0))

        padded_images.append(new_img)

    return padded_images

class Collator:
    def __init__(self, tokenizer):
        # self.tokenizer = tokenizer
        self.transforms_ = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),  # Resize to height 64, width 128
                transforms.ToTensor(),         # Convert PIL Image to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
        self.tokenizer = tokenizer

    def collate_fn(self, batch):

        # (NUM_IMAGES, 3, 224, 224)
        images = sum([x['images'] for x in batch], start=[])
        images = resize_and_pad_images(images, 64, 16)

        #print([im.size for im in images])
        images = torch.stack([self.transforms_(x) for x in images])
        images_aux_batchs = [len(desc['images']) for batch, desc in enumerate(batch)]
        images_batch_idxs =[[sum(images_aux_batchs[:batch]) + i for i in range(len(desc['images']))]
                         for batch, desc in enumerate(batch)]

        # (NUM_QUERIES, 77)

        queries = sum([x['query'] for x in batch], start=[])
        padding_to = len(max(queries, key=lambda x: len(x)))
        queries = torch.tensor([que + [0] * (padding_to - len(que)) for que in queries]).view(-1, padding_to)

        queries_aux_batchs = [len(desc['query']) for batch, desc in enumerate(batch)]
        queries_batch_idxs =[[sum(queries_aux_batchs[:batch]) + i for i in range(len(desc['query']))]
                         for batch, desc in enumerate(batch)]

        return {
            'images': images,
            'images_batch_idxs': images_batch_idxs,
            'queries': queries,
            'queries_batch_idxs': queries_batch_idxs,
            'ocr_words': [x['words'] for x in batch],
            'query_str': [x['query_str'] for x in batch]
        }




