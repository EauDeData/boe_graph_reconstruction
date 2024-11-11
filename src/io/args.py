import torch

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Argument Parser for your script')

    # Dataset arguments
    parser.add_argument('--output_name', type=str, default=None, help='Model output name (leave empty for none)')
    parser.add_argument('--vision_model', type=str, default='cnn', help='Model output name (leave empty for none)')

    parser.add_argument('--model_ckpt_name', type=str, default=None, help='Model ckpt name (leave empty for none)')
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--local_p_loss', type=float, default=.25)
    parser.add_argument('--global_p_loss', type=float, default=.25)
    parser.add_argument('--topic_p_loss', type=float, default=.5)


    # parser.add_argument('--imsize', type=int, default=224, help='Image size for resizing')
    # parser.add_argument('--seq_max_leng', type=int, default=77, help='Maximum sequence length for tokenization')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')

    # Model arguments
    parser.add_argument('--tokens_dim', type=int, default=128, help='Size of the text embeddings')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run the models on')

    parser.add_argument('--visual_lr', type=float, default=1e-6, help='Learning rate for the optimizer')

    parser.add_argument('--log_wandb', action='store_true', help='Whether to log using WandB')
    parser.add_argument('--epoches', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--td', default=.5, type=float)
    parser.add_argument('--ti', default=.5, type=float)

    return parser.parse_args()
