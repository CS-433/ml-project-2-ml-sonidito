from tqdm.autonotebook import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

def train(model,train_loader, val_loader, optimizer, criterion, device,
            n_epochs = 10, early_stopping = None, l1_sigma=0):

    model.to(device)

    train_losses = []
    val_losses = []

    val_losses.append(run_validation(model, val_loader, criterion, device))
    
    for epoch in range(n_epochs):
        epoch_losses = train_one_epoch(model, train_loader, criterion, optimizer, epoch, l1_sigma, device)
        train_losses += epoch_losses

        eval_loss = run_validation(model, val_loader, criterion, device)
        val_losses.append(eval_loss)

        if early_stopping != None:
            if early_stopping.early_stop(eval_loss):
                break
    
    plot_losses(train_losses, val_losses)

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, l1_sigma, device):
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
            if l1_sigma != 0 :
                for param in model.parameters():
                    loss = loss + l1_sigma * torch.norm(param, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            tepoch.set_postfix(loss=np.mean(batch_losses))

    return batch_losses

def run_validation(model, val_loader, criterion, device) :
    model.eval()    
    running_val_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader :
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            y_batch = y_batch.to(torch.float32)
            loss = criterion(pred, y_batch)

            running_val_loss += loss.item()

    return running_val_loss / len(val_loader)

def plot_losses(train_losses, val_losses):
    plt.plot(np.linspace(0, len(val_losses)-1, len(train_losses)), train_losses, label='train')
    plt.plot(val_losses, label='validation')
    plt.title("loss")
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.legend()