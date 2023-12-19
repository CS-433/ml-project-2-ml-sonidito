from tqdm.notebook import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from metrics import *

import torch.optim as optim
import torch.nn as nn


def train(model, train_loader, test_loader, optimizer, scheduler, criterion, device, n_epochs=1):
    model.to(device)
    print(f"training on device '{device}'")

    train_losses = []
    val_losses = []
    f1s_test = []
    f1s_train = []

    for epoch in range(n_epochs):

        model.train()
        with tqdm(train_loader, unit='batch', desc=f'Epoch {epoch}') as tepoch:
            for batch in tepoch:
                x_batch = batch['window_odd']
                y_batch = batch['label']
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                output = model(x_batch)

                loss = criterion(output, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                tepoch.set_postfix(loss=loss.item())


        model.eval()
        # evaluate on validation set
        val_acc = []
        val_f1 = []
        val_roc_auc =[]
        val_loss = 0
        with tqdm(test_loader, unit='batch', desc='Evaluating') as tepoch:
            for batch in tepoch:
                x_batch = batch['window_odd']
                y_batch = batch['label']
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                with torch.no_grad():
                    loss = criterion(model(x_batch), y_batch)
                    acc, f1, roc_auc = compute_metrics(model(x_batch), y_batch)

                    val_loss += loss.item()
                    val_acc.append(acc)
                    val_f1.append(f1)
                    val_roc_auc.append(roc_auc)


                tepoch.set_postfix(loss=loss.item())

        val_losses.append(val_loss / len(test_loader))
        f1s_test.append(np.mean(val_f1))

        # evaluate on train set
        train_acc = []
        train_f1 = []
        train_roc_auc = []
        train_loss = 0
        with tqdm(train_loader, unit='batch', desc='Evaluating') as tepoch:
            for batch in tepoch:
                x_batch = batch['window_odd']
                y_batch = batch['label']
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                with torch.no_grad():
                    output = model(x_batch)
                    loss = criterion(output, y_batch)
                    acc, f1, roc_auc = compute_metrics(output, y_batch)

                    train_loss += loss.item()
                    train_acc.append(acc)
                    train_f1.append(f1)
                    train_roc_auc.append(roc_auc)

                tepoch.set_postfix(loss=loss.item())

        train_losses.append(train_loss / len(train_loader))
        f1s_train.append(np.mean(train_f1))


        train_roc_auc = [x for x in train_roc_auc if x is not None]
        val_roc_auc = [x for x in val_roc_auc if x is not None]
        print(
            f"Epoch {epoch} | Train accuracy: {torch.tensor(train_acc).mean().item():.5f}, "
            f"f1: {torch.tensor(train_f1).mean().item():.5f}, "
            f"roc-auc: {torch.tensor(train_roc_auc).mean().item():.5f}\n "
            f"           Test accuracy: {torch.tensor(val_acc).mean().item():.5f}, "
            f"f1: {torch.tensor(val_f1).mean().item():.5f}, "
            f"roc-auc: {torch.tensor(val_roc_auc).mean().item():.5f}"
        )

    plot_losses(train_losses, val_losses)
    plot_f1(f1s_train, f1s_test)

    return f1s_train[-1], f1s_test[-1]


def plot_losses(losses, losses_val):
    plt.plot(losses, label='train')
    plt.plot(losses_val, label='test')
    plt.title("loss")
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.legend()
    plt.show()


def plot_f1(f1s_train, f1s_test):
    plt.plot(f1s_train, label='train')
    plt.plot(f1s_test, label='test')
    plt.title("f1")
    plt.ylabel('f1')
    plt.xlabel('iterations')
    plt.legend()
    plt.show()