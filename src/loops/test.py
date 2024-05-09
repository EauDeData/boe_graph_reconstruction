from tqdm import tqdm
import torch
def eval_step(joint_model, dataloader, optimizer, loss_f, logger, epoch ):

    total_loss = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total = len(dataloader)):

            node_features = joint_model(batch)

            loss = loss_f(node_features['regressed_features'], batch['gt'].to(joint_model.device))

            total_loss.append(loss)

    return total_loss