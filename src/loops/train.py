from tqdm import tqdm
def train_step(joint_model, dataloader, optimizer, loss_f, logger, epoch ):

    total_loss = []
    for batch in tqdm(dataloader, total = len(dataloader)):

        optimizer.zero_grad()

        node_features = joint_model(batch)

        loss = loss_f(node_features['document'], node_features['queries'])
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
        # if logger:
        #     logger.log({
        #         'loss': loss.item(),
        #         'epoch': epoch
        #     })
    return total_loss
