from tqdm import tqdm
def train_step(joint_model, dataloader, optimizer, loss_f, logger, epoch ):

    total_loss = []
    joint_model.train()
    for batch in tqdm(dataloader, total = len(dataloader)):

        optimizer.zero_grad()

        loss = joint_model(batch)

        loss.backward()
        optimizer.step()

        total_loss.append(loss)
        if logger:
            logger.log({
                'loss': loss.item(),
                'epoch': epoch
            })
    return total_loss
