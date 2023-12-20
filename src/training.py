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
    train_f1s = []
    val_f1s = []
    train_kappas = []
    val_kappas = []
    train_best_treshs = []
    val_best_treshs = []

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
        val_logits = []
        val_labels = []
        val_loss = 0
        with tqdm(test_loader, unit='batch', desc='Evaluating') as tepoch:
            for batch in tepoch:
                x_batch = batch['window_odd']
                y_batch = batch['label']
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                with torch.no_grad():
                    output = model(x_batch)
                    loss = criterion(output, y_batch)
                    val_logits.append(output)
                    val_labels.append(y_batch)

                    val_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())

        val_losses.append(val_loss / len(test_loader))
        # compute metrics over entire epoch
        val_acc, val_f1, val_kappa, val_best_tresh = compute_metrics(torch.cat(val_logits, dim=0),
                                                                     torch.cat(val_labels, dim=0))
        val_f1s.append(val_f1)
        val_kappas.append(val_kappa)
        val_best_treshs.append(val_best_tresh)

        # evaluate on train set
        train_logits = []
        train_labels = []
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
                    train_logits.append(output)
                    train_labels.append(y_batch)

                    train_loss += loss.item()

                tepoch.set_postfix(loss=loss.item())

        train_losses.append(train_loss / len(train_loader))
        # compute metrics over entire epoch
        train_acc, train_f1, train_kappa, train_best_tresh = compute_metrics(torch.cat(train_logits, dim=0),
                                                                             torch.cat(train_labels, dim=0))
        train_f1s.append(train_f1)
        train_kappas.append(train_kappa)
        train_best_treshs.append(train_best_tresh)

        print(
            f"Epoch {epoch} | Train accuracy: {train_acc.item():.3f}, "
            f"f1: {train_f1.item():.3f}, "
            f"kappa: {train_kappa.item():.3f}, "
            f"best threshold: {train_best_tresh.item():.3f}\n "
            f"    Validation accuracy: {val_acc.item():.3f}, "
            f"f1: {val_f1.item():.3f}, "
            f"kappa: {val_kappa.item():.3f}, "
            f"best threshold: {val_best_tresh.item():.3f}"
        )

    plot_metric(train_losses, val_losses, 'loss')
    plot_metric(train_f1s, val_f1s, 'f1')
    plot_metric(train_kappas, val_kappas, 'kappa')
    print(f'best threshold over validation set: {val_best_treshs[-1]}')

    return train_f1s[-1], val_f1s[-1], train_kappas[-1], val_kappas[-1], train_best_treshs[-1], val_best_treshs[-1]


def plot_metric(train_metric, val_metric, metric_name):
    plt.plot(train_metric, label='train')
    plt.plot(val_metric, label='validation')
    plt.title(metric_name)
    plt.ylabel(metric_name)
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
