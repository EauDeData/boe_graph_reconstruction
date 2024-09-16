from tqdm import tqdm
def train_step(joint_model, dataloader, optimizer, loss_f, logger, epoch ):

    total_loss = []
    joint_model.train()
    for batch in tqdm(dataloader, total = len(dataloader)):

        optimizer.zero_grad()
        exit()
        loss = joint_model(batch)
        if logger:
            logger.log(loss)

        # loss = loss['bipartite_loss'] + loss['context_loss']
        loss = loss['context_loss']
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
        if logger:
            logger.log({
                'loss': loss.item(),
                'epoch': epoch
            })
    return total_loss
