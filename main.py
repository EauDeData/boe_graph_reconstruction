import os

import torch

from src.data.datasets import BOEDataset
from src.data.collator import Collator
from src.tokenizer.text import BERTTokenizer
from src.data.defaults import IMAGENET_STDS, IMAGENET_MEANS, OUTPUT_FOLDER_NAME
from src.models.text import TransformerTextEncoder
from src.models.graphs import GraphConv
from src.models.utils import JointModel
from src.loops.train import train_step
from src.loops.test import eval_step
import json

from torch.utils.data import DataLoader
from torch.nn import MSELoss, DataParallel
from torch.optim import AdamW, Adam

import torchvision
import open_clip

def load_datasets(args):

    train_set = BOEDataset('train.txt')
    test_set = BOEDataset('test.txt')

    collator = Collator(train_set.tokenizer)
    return [DataLoader(data,
                       drop_last=True,
                       collate_fn=collator.collate_fn,
                       num_workers=args.num_workers,
                       batch_size=args.batch_size,
                       shuffle=[True, False][n]) for n, data in enumerate((train_set, test_set))], collator

# def load_models(args, collator):
#     num_tokens = len(collator.tokenizer)
#     args.text_emb_size += args.text_emb_size % args.num_text_heads
#     text_model = TransformerTextEncoder(num_tokens, args.text_emb_size, args.num_text_heads, args.num_text_layers,device=args.device).to(args.device)
#     queries_model = TransformerTextEncoder(num_tokens, args.graph_out_channels * 2, args.num_text_heads, args.num_text_layers,device=args.device).to(args.device)
#     visual_model = CLIPVisionEncoder(device=args.device).to(args.device)
#     # When using clip, it is 768. In the future it will be not hardcoded
#     graph_model = GraphConv(args.text_emb_size, 768, args.graph_in_channels,
#                             args.graph_hidden_channels, args.graph_depth, args.graph_out_channels, num_categories=2,
#                             device=args.device).to(args.device)
#
#     return text_model, visual_model, graph_model, queries_model
def get_learning_rates(optimizer):
    lr_dict = {}
    for i, param_group in enumerate(optimizer.param_groups):
        lr_dict[f'param_group_{i}'] = param_group['lr']
    return lr_dict
def main(args):

    os.makedirs(OUTPUT_FOLDER_NAME, exist_ok=True)
    (train_loader, test_loader), collator = load_datasets(args)
    # text_model, visual_model, graph_model, queries_model = load_models(args, collator)
    # joint_model = JointModel(visual_model, text_model, graph_model, queries_model)
    joint_model = JointModel(len(train_loader.dataset.tokenizer), args.tokens_dim,
                             margin=args.margin, local_p = args.local_p_loss,
                             global_p=args.global_p_loss, topic_p=args.topic_p_loss,
                             device=args.device, args=args)

    if isinstance(args.model_ckpt_name, str):
        joint_model.load_state_dict(torch.load(args.model_ckpt_name))

    optimizer = Adam(joint_model.parameters(), lr=args.visual_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    if args.log_wandb:
        import wandb
        wandb.init(project='boe_graph_reconstruction')
        wandb.config.update(args)
        logger = wandb
    else:
        logger = None

    losses = eval_step(joint_model, test_loader, optimizer, None, logger, 0)

    if logger:
        logger.log(losses)
    print("Random baseline tested with avg loss:", losses)

    for epoch in range(1, args.epoches):
        if logger:
            logger.log(get_learning_rates(optimizer))

        if not args.output_name is None:
            torch.save(joint_model.state_dict(), os.path.join(OUTPUT_FOLDER_NAME, args.output_name + '.pth'))
            json.dump(vars(args), open(os.path.join(OUTPUT_FOLDER_NAME, args.output_name + '.json'), 'w'))

        print(f"Training epoch {epoch} out {args.epoches}")
        losses = train_step(joint_model, train_loader, optimizer, None, logger, epoch)
        scheduler.step(sum(losses) / len(losses))
        if logger:
            logger.log({'epoch_loss': sum(losses) / len(losses)})
        print("Trained with avg loss:", sum(losses) / len(losses))

        print(f"Testing epoch {epoch} out {args.epoches}")
        losses = eval_step(joint_model, test_loader, optimizer, None, logger, epoch)
        if logger:
            logger.log(losses)
        print("Tested with avg loss:", losses)



if __name__ == '__main__':
    from src.io.args import parse_args
    main(parse_args())
