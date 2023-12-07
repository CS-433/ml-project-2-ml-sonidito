from tqdm.autonotebook import tqdm
import torch
import matplotlib.pyplot as plt

def train(model,train_loader, val_loader, optimizer, criterion, device, n_epochs = 10, early_stopping = None):

    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, device)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0
    
        with torch.no_grad():
            for x_batch, y_batch in val_loader :
                pred = model(x_batch)
                y_batch = y_batch.to(torch.float32)
                loss = criterion(pred, y_batch)

                running_val_loss += loss.item()
            val_losses.append(running_val_loss / len(val_loader))

        if early_stopping != None:
            if early_stopping.early_stop(running_val_loss / len(val_loader)):
                break
    
    plot_losses(train_losses, val_losses)

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_train_loss = 0
    with tqdm(train_loader, unit='batch') as tepoch:
        for x_batch, y_batch in tepoch :
            tepoch.set_description(f'Epoch {epoch}')

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            y_batch = y_batch.to(torch.float32)

            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            tepoch.set_postfix(loss=loss.item())

            running_train_loss += loss.item()

    avg =  running_train_loss / len(train_loader)
    tepoch.set_postfix(loss=avg)
    return avg

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='validation')
    plt.title("loss")
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.legend()