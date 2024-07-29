import networkx as nx
import numpy as np
import torch
from PIL import  Image


def resize_and_pad_images(images, fixed_height, patch_size, max_allowed_width = 128):
    resized_images = []
    max_width = 0

    # Resize images to fixed height and width multiple of patch_size
    for img in images:
        # aspect_ratio = img.width / img.height
        width, height = img.size

        # new_width = int(aspect_ratio * fixed_height)
        # Adjust new_width to be a multiple of patch_size
        new_width = new_width = (width + (patch_size - 1)) // patch_size * patch_size
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
    def __init__(self, transforms):
        # self.tokenizer = tokenizer
        self.transforms = transforms
        self.transforms.transforms = self.transforms.transforms[2:]

    def collate_fn(self, batch):

        # (NUM_IMAGES, 3, 224, 224)
        images = sum([x['images'] for x in batch], start=[])
        images = resize_and_pad_images(images, 64, 16)

        #print([im.size for im in images])
        images = torch.stack([self.transforms(x) for x in images])
        images_aux_batchs = [len(desc['images']) for batch, desc in enumerate(batch)]
        images_batch_idxs =[[sum(images_aux_batchs[:batch]) + i for i in range(len(desc['images']))]
                         for batch, desc in enumerate(batch)]

        # (NUM_QUERIES, 77)
        queries = torch.stack(sum([x['query'] for x in batch], start=[])).view(-1, 77)
        queries_aux_batchs = [len(desc['query']) for batch, desc in enumerate(batch)]
        queries_batch_idxs =[[sum(queries_aux_batchs[:batch]) + i for i in range(len(desc['query']))]
                         for batch, desc in enumerate(batch)]

        # dates = torch.stack([x['phoc_dates'] for x in batch])

        # ocr_phoc = torch.cat([x['phoc_ocr'] for x in batch], dim=0)
        # query_phoc = torch.cat([x['phoc_query'] for x in batch], dim=0)
        #
        #
        # ocr_aux_batchs = [desc['phoc_ocr'].shape[0] for batch, desc in enumerate(batch)]
        # ocr_batch_idxs =[[sum(ocr_aux_batchs[:batch]) + i for i in range(desc['phoc_ocr'].shape[0])]
        #                  for batch, desc in enumerate(batch)]
        #
        # queries_aux_batchs = [desc['phoc_query'].shape[0] for batch, desc in enumerate(batch)]


        return {
            'images': images,
            'images_batch_idxs': images_batch_idxs,
            'queries': queries,
            'queries_batch_idxs': queries_batch_idxs
            # 'phocked_ocr': ocr_phoc,
            # 'ocrs_batch_idxs': ocr_batch_idxs,
            # 'phocked_queries': query_phoc,
            # 'queries_batch_idxs': queries_batch_idx,
            # # 'dates': dates,
            # # 'images': images
        }




