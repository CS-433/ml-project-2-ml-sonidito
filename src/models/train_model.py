from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

def train(model,train_loader, optimizer, criterion, device, n_epochs = 10):
    model.train()

    losses = []

    for epoch in tqdm(range(n_epochs)):
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch).squeeze()
            y_batch = y_batch.to(torch.float32)

            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % (len(train_loader.dataset) // len(train_loader) // 10) == 0:
                print(
                    f"Train Epoch: {epoch}-{batch_idx:03d} "
                    f"batch_loss={loss.item():0.2e} "
                )
                losses.append(loss.item())
            # TODO : add eval
    
    plot_losses(losses)

def plot_losses(losses):
    plt.plot(losses, label='train')
    plt.title("loss")
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.legend()