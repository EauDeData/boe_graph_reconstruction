import torch

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Argument Parser for your script')

    # Dataset arguments
    parser.add_argument('--imsize', type=int, default=224, help='Image size for resizing')
    parser.add_argument('--seq_max_leng', type=int, default=77, help='Maximum sequence length for tokenization')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')

    # Model arguments
    parser.add_argument('--text_emb_size', type=int, default=512, help='Size of the text embeddings')
    parser.add_argument('--num_text_heads', type=int, default=4, help='Number of attention heads in the text model')
    parser.add_argument('--num_text_layers', type=int, default=2, help='Number of layers in the text model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run the models on')

    # Graph model arguments
    parser.add_argument('--graph_in_channels', type=int, default=256, help='Input channels for the graph model')
    parser.add_argument('--graph_hidden_channels', type=int, default=256, help='Hidden channels for the graph model')
    parser.add_argument('--graph_depth', type=int, default=3, help='Depth of the graph model')
    parser.add_argument('--graph_out_channels', type=int, default=256, help='Output channels of the graph model')

    # Training arguments
    parser.add_argument('--visual_lr', type=float, default=1e-6, help='Learning rate for the optimizer')
    parser.add_argument('--textual_lr', type=float, default=1e-5, help='Learning rate for the optimizer')
    parser.add_argument('--graph_lr', type=float, default=1e-4, help='Learning rate for the optimizer')

    parser.add_argument('--log_wandb', action='store_true', help='Whether to log using WandB')
    parser.add_argument('--epoches', type=int, default=10, help='Number of epochs for training')

    return parser.parse_args()
