from tqdm.autonotebook import tqdm
import torch
import matplotlib.pyplot as plt

def train(model,train_loader, optimizer, criterion, device, n_epochs = 10):
    model.train()

    losses = []

    for epoch in range(n_epochs):
        with tqdm(train_loader, unit='batch') as tepoch:
            for x_batch, y_batch in tepoch :
                tepoch.set_description(f'Epoch {epoch}')

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(x_batch).squeeze()
                y_batch = y_batch.to(torch.float32)

                loss = criterion(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                tepoch.set_postfix(loss=loss.item())
               
                losses.append(loss.item())
                # TODO : add eval
    
    plot_losses(losses)

def plot_losses(losses):
    plt.plot(losses, label='train')
    plt.title("loss")
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.legend()