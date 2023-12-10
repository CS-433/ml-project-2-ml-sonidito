from tqdm.autonotebook import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

def train(model,train_loader, val_loader, optimizer, criterion, device, n_epochs = 10, early_stopping = None):

    train_losses = []
    val_losses = []

    val_losses.append(run_validation(model, val_loader, criterion))
    
    for epoch in range(n_epochs):
        epoch_losses = train_one_epoch(model, train_loader, criterion, optimizer, epoch, device)
        train_losses += epoch_losses

        eval_loss = run_validation(model, val_loader, criterion)
        val_losses.append(eval_loss)

        if early_stopping != None:
            if early_stopping.early_stop(eval_loss):
                break
    
    plot_losses(train_losses, val_losses, n_epochs)

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    batch_losses = []
    with tqdm(train_loader, unit='batch') as tepoch:
        for x_batch, y_batch in tepoch :
            tepoch.set_description(f'Epoch {epoch+1}')

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            y_batch = y_batch.to(torch.float32)

            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            tepoch.set_postfix(loss=np.mean(batch_losses))

    return batch_losses

def run_validation(model, val_loader, criterion) :
    model.eval()    
    running_val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader :
            pred = model(x_batch)
            y_batch = y_batch.to(torch.float32)
            loss = criterion(pred, y_batch)

            running_val_loss += loss.item()

    return running_val_loss / len(val_loader)

def plot_losses(train_losses, val_losses, n_epochs):
    plt.plot(np.linspace(0, n_epochs, len(train_losses)), train_losses, label='train')
    plt.plot(val_losses, label='validation')
    plt.title("loss")
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.legend()