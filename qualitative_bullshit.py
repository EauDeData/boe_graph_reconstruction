import torch
from torch.utils.data import DataLoader
from src.data.datasets import BOEDataset
from src.data.collator import Collator
from src.models.utils import JointModel
from src.models.soft_hungarian import  SoftHdSimetric, SoftHd
import torch.nn as nn
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
from src.data.defaults import IMAGE_SIZE
from src.loops.test import eval_step

def standarize(image):
    return (image - image.min()) / (image.max() - image.min())


def visualize(batch, doc_to_query_corr, query_to_doc_corr):
    images = [standarize(x.numpy().transpose(1, 2, 0)) for x in batch['images_document'].squeeze()]

    # the [1:] avoids the CLS
    token_keys = [standarize(x.numpy().transpose(1, 2, 0)) for x in batch[
        'images_query'].squeeze()]  # [tokens[idx] for idx in batch['query_text_tokens'].squeeze().numpy()[1:]]

    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 32

    # PLOTTING TIME
    # Create the plot
    fig, ax = plt.subplots(figsize=[x // 4 for x in IMAGE_SIZE])
    num_images = len(images)
    num_token_keys = len(token_keys)
    # Plot images in the left column
    # Plot images in the left column
    # Scaling factor to respect aspect ratio in the plot (since 32x128 is taller than wide)
    aspect_ratio = IMAGE_HEIGHT / IMAGE_WIDTH
    vertical_scale = 1.0  # Use this scale to keep the distance between rows visible.

    # Plot images in the left column with rectangular shape (32x128)
    for i in range(len(images)):
        # Calculate the extent so that it respects the aspect ratio (placing by center)
        y_center_left = num_images - i - 0.5
        ax.imshow(images[i], extent=[0, 1, y_center_left - 0.5 * vertical_scale * aspect_ratio,
                                     y_center_left + 0.5 * vertical_scale * aspect_ratio], cmap='gray')
        # If the matching index is infinity, draw a red frame around this image
        if doc_to_query_corr[i] == np.inf:
            rect_left = plt.Rectangle((0, y_center_left - 0.5 * vertical_scale * aspect_ratio),
                                      1, vertical_scale * aspect_ratio, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect_left)

    # Plot images in the right column with rectangular shape (32x128)
    for i in range(len(token_keys)):
        # Calculate the extent for right column images
        y_center_right = num_token_keys - i - 0.5
        ax.imshow(token_keys[i], extent=[3, 4, y_center_right - 0.5 * vertical_scale * aspect_ratio,
                                         y_center_right + 0.5 * vertical_scale * aspect_ratio], cmap='gray')

        # If the matching index is infinity, draw a red frame around this image
        # if query_to_doc_corr[i] == np.inf:
        #     rect_right = plt.Rectangle((3, y_center_right - 0.5 * vertical_scale * aspect_ratio),
        #                                1, vertical_scale * aspect_ratio, linewidth=2, edgecolor='red', facecolor='none')
        #     ax.add_patch(rect_right)

    # Draw matching lines from the center of left images to the center of the right images (doc_to_query_corr)
    for i in range(len(images)):
        image_idx = i  # index from the left column (images)
        word_idx = doc_to_query_corr[i]  # corresponding index from the right column (token_keys)

        # Calculate the y-center for both the left and right images
        y_center_left = num_images - image_idx - 0.5
        y_center_right = num_token_keys - word_idx - 0.5

        # Draw the blue line connecting the centers of the corresponding images (left to right)
        ax.plot([1, 3], [y_center_left, y_center_right], color="blue")

    # Draw matching lines from the right column images to the left column (query_to_doc_corr)
    for i in range(len(token_keys)):
        word_idx = i  # index from the right column (token_keys)
        image_idx = query_to_doc_corr[i]  # corresponding index from the left column (images)

        # Calculate the y-center for both the right and left images
        y_center_right = num_token_keys - word_idx - 0.5
        y_center_left = num_images - image_idx - 0.5

        # Draw the green line connecting the centers of the corresponding images (right to left)
        ax.plot([3, 1], [y_center_right, y_center_left], color="green")

    # Adjust plot settings
    # ax.set_xlim(0, 5)
    # ax.set_ylim(0, max(num_images, num_token_keys))  # Ensure y-limit covers the taller column
    ax.axis("off")

    # Show the plot
    plt.savefig('tmp4.png')

def get_corrs_from_batch(batch, model):

    # Get the features from the batch
    ((dense_doc_features, doc_features_mask), (dense_query_features, query_features_mask),
     query_tokens_embedding) = model.common_process(batch)

    t1_features, t2_features = dense_doc_features.squeeze(), dense_query_features.squeeze()

    print(t1_features.shape, t2_features.shape)
    t1_edges = model.hungarian_distance.produce_reading_order_edges(t1_features.shape[0]).to(model.device)
    t2_edges = model.hungarian_distance.produce_reading_order_edges(t2_features.shape[0]).to(model.device)
    print(t1_edges, t2_edges)

    message_passed_t1 =  model.hungarian_distance.gat(
        x=t1_features,
        edge_index=t1_edges
    )

    message_passed_t2 =  model.hungarian_distance.gat(
        x=t2_features,
        edge_index=t2_edges
    )

    _, corr_AB, corr_BA =  model.hungarian_distance.haussdorf_distance(t1_features, t2_features,
                                                                       message_passed_t1, message_passed_t2,
                                                                       inference=True)

    return corr_AB, corr_AB



distractor_set = BOEDataset('test.txt')
# Haurem de tallar els distractors
# distractor_set.crop_dataset()

collator = Collator(distractor_set.tokenizer)


distractor_dataloader = DataLoader(distractor_set,
                       collate_fn=collator.collate_fn,
                       num_workers=4,
                       batch_size=1,
                       shuffle=False) # Molt important el no shuffle!!!
class Args:
    td = 0.5
    ti = 0.5
    def __init__(self):
        pass
joint_model = JointModel(len(distractor_dataloader.dataset.tokenizer),
                         128, 0.1, 0.5, 0.25, 0.5, device='cpu',
                         hung_constructor=SoftHd, args=Args())

joint_model.load_state_dict(torch.load('/data/users/amolina/leviatan/structured_approaches/asmetric_matching_delete_pre_passing_this_should_work.pth'))
joint_model.eval()
for batch in distractor_dataloader:
        visualize(batch, *get_corrs_from_batch(batch, joint_model))
        exit()
# print(eval_step(joint_model, distractor_dataloader, None, None, None, None, False))
