from tqdm.autonotebook import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os

from models.metrics import *
from models.early_stopping import EarlyStopping

from models.simple_LSTM import SimpleLSTM
import torch.optim as optim
import torch.nn as nn


def train(model, train_loader, val_loader, optimizer, criterion, device, 
          n_epochs = 10, l1_sigma=0, compute_objective = None,
          direction="minimize" , patience=0, delta=1e-5, model_path="models", save_plots=False, plot_folder=None):
    
    """
        Helper used to train the given model and plot the losses after the training

        Parameters 
            model : nn.Module
                Model to train
            train_loader : Dataloader
                Data used for the training
            val_loader : Dataloader
                Data used for th validation
            optimizer : nn.Optim
                Optimzer used during the training
            criterion : torch.nn
                criterion used to compute the loss
            device : str
                Used to move the tensor to the correct device
            n_epochs : int, optional
                Number of epoch to run
            early_stopping : EarlyStopping, optional
                Module used to track the early stop condition and save the best model
            l1_sigma : float, optional
                Sigma used of the L1 regularization
            compute_objective : func, optional
                Function used to compute the score using the validation data
            direction : str, optional
                objective direction, can be minimize or maximize
    """

    model.to(device)

    train_losses = []
    val_losses = []

    scores_list = []

    early_stopping = None
    
    if patience != 0:
        early_stopping = EarlyStopping(model_path, patience, delta, direction)


    eval_loss, scores = run_validation(model, val_loader, criterion, device, compute_objective)
    val_losses.append(eval_loss)
    if compute_objective is not None:
        scores = torch.mean(torch.tensor(scores))
        scores_list.append(scores.item())

    best_score = float('inf')

    if direction == "maximize":
        best_score = -float('inf')

    with tqdm(range(n_epochs)) as pbar: 
        for epoch in pbar:
            epoch_losses = train_one_epoch(model, 
                                           train_loader, 
                                           criterion, 
                                           optimizer, 
                                           epoch, 
                                           l1_sigma, 
                                           device, 
                                           leave=False)
            train_losses += epoch_losses

            eval_loss, scores = run_validation(model, val_loader, criterion, device, compute_objective)
            val_losses.append(eval_loss)
            if compute_objective is not None :
                scores = torch.mean(torch.tensor(scores))
                scores_list.append(scores.item())
            else :
                scores = eval_loss

            pbar.set_postfix(score=scores.item(), train_loss=np.mean(train_losses))

            if direction == "minimize" : 
                if scores < best_score:
                    best_score = scores
                    torch.save(model, model_path)
            elif direction == "maximize" : 
                if scores > best_score:
                    best_score = scores
                    torch.save(model, model_path)

            if early_stopping != None:
                if early_stopping.early_stop(scores, model):
                    break

    plot_losses(train_losses, val_losses, save_plots, plot_folder)

    if compute_objective  is not None:
        plot_scores(scores_list, save_plots, plot_folder)

    model = torch.load(model_path) # laod best model
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, l1_sigma, device, disable_progress_bar=False, leave=True):
    """
        Train one epoch

        Parameter
            model : nn.Module
                Model to train
            train_loader : DataLoader
                Data used for the training
            criterion : torch.nn
                Criterion used to compute the loss
            epoch : int
                Current epoch
            l1_sigma : float
                Sigma of the L1 regularization
            device : str
                Used to move the tensor to the correct device
            disable_progress_bar : bool, optional
                if True, disable tqdm progress bar
            leave : bool, optional
                leave option of tqdm
        
        Return
            list
                List of batch loss
    """
    model.train()
    batch_losses = []
    with tqdm(train_loader, unit='batch', disable=disable_progress_bar, leave=leave) as tepoch:
        for (x_batch, metadata), y_batch in tepoch :
            tepoch.set_description(f'Epoch {epoch+1}')

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            y_batch = y_batch.to(torch.float32)

            loss = criterion(y_pred, y_batch)

            # L1 Normalization
            if l1_sigma != 0 :
                for param in model.parameters():
                    if param.requires_grad:
                        loss = loss + l1_sigma * torch.norm(param, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            tepoch.set_postfix(loss=np.mean(batch_losses))

    return batch_losses

def run_validation(model, val_loader, criterion, device, compute_objective = None) :
    """"
        Compute the score and the loss of the validation data

        Parameter
            model : nn.Module
                Model to evaluate
            val_loader : Dataloader
                Data used for th validation
            criterion : torch.nn
                criterion used to compute the loss
            device : str
                Used to move the tensor to the correct device
            compute_objective : func, optional
                Function used to compute the score using the validation data
        
        Return 
            float
                the mean loss of the validation
            float
                the mean score of the validation
    """

    model.eval()    
    running_val_loss = 0
    scores = []

    with torch.no_grad():
        for (x_batch, metadata), y_batch in val_loader :

            x_batch = x_batch.to(device)

            output = model(x_batch)
            loss = criterion(output, y_batch.to(device))

            if compute_objective is not None:
                scores += compute_objective(output, y_batch)

            running_val_loss += loss.item()

    return running_val_loss / len(val_loader), scores


def k_fold(dataset, model, criterion, device, lr, weight_decay, create_output,
           k_fold=5, n_epochs = 1, l1_sigma=0, batch_size=1, threshold=0.8, disable_pbar=False) :
    
    """
        Run K-fold cross validation

        Parameter 
            dataset : Dataset
                Dataset used for the cross validation
            model : nn.Module
                Model to evaluate
            criterion : torch.nn
                criterion used to compute the loss
            lr : float
                Learning rate
            weigth_decay : float
                weight decay for the optimizer
            create_output : func
                Function used to compute the score
            k_fold : int, optional
                number of fold
            n_epochs : int, optional
                Number of epoch to run
            l1_sigma : float, optional
                Sigma used of the L1 regularization
            batch_size : int, optional
                Batch size of the dataloader
            threshold : float, optional
                Threshold used by the `create_output` function
            disable_pbar : bool, optional 
                remove tqdm bar
        
        Return 
            dict
                the average of different score
                Score computed : accuracy, precision, recall, f1, kappa, train_loss, val_loss
    """
        
    model.to(device)

    kd = KFold(k_fold, shuffle=True)

    mean_accuracy = 0
    mean_precision = 0
    mean_recall = 0
    mean_f1 = 0
    mean_train_loss = 0
    mean_eval_loss = 0
    mean_kappa = 0
    

    for train_ids, test_ids in tqdm(kd.split(dataset), unit='fold', disable=disable_pbar, total=k_fold):

        # Create the split
        train_subsample = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsample = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsample)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_subsample)
    
        # Reset the model weights
        model.reset_weights()

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            epoch_losses = train_one_epoch(model, 
                                           train_loader, 
                                           criterion, 
                                           optimizer, 
                                           epoch, 
                                           l1_sigma, 
                                           device, 
                                           disable_progress_bar=True)
            train_losses += epoch_losses

            eval_loss, _ = run_validation(model, val_loader, criterion, device)
            val_losses.append(eval_loss)
        
        mean_train_loss += np.mean(train_losses) / k_fold
        mean_eval_loss += np.mean(val_losses) / k_fold

        # Model eval
        model.eval()
        with torch.no_grad():
            total_accuracy = 0
            total_precision = 0
            total_recall = 0
            total_f1 = 0
            total_kappa = 0

            count = 0

            for (x_batch, metadata), y_batch in val_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                output = model(x_batch)
                logits = torch.sigmoid(output)

                preds = create_output(logits, threshold).to(device)

                total_accuracy += torch.sum(compute_accuracy(preds, y_batch)) 
                total_precision += torch.sum(compute_precision(preds, y_batch)) 
                total_recall += torch.sum(compute_recall(preds, y_batch)) 
                total_f1 += torch.sum(compute_f1(preds, y_batch))
                total_kappa += torch.sum(compute_kappa_score(preds, y_batch))

                count += x_batch.shape[0]

            mean_accuracy += total_accuracy / k_fold / count
            mean_precision += total_precision / k_fold / count
            mean_recall += total_recall / k_fold / count
            mean_f1 += total_f1 / k_fold / count
            mean_kappa += total_kappa / k_fold / count
    
    return {'accuracy': mean_accuracy.item(),
            'precision' : mean_precision.item(),
            'recall' : mean_recall.item(),
            'f1' : mean_f1.item(),
            'kappa' : mean_kappa.item(),
            'train_loss' : mean_train_loss,
            'val_loss' : mean_eval_loss}


def plot_losses(train_losses, val_losses, save_plots, plot_folder):
    """
        Plot the train and validation on the same plot
    """
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, len(val_losses)-1, len(train_losses)), train_losses, label='train')
    ax.plot(val_losses, label='validation')
    ax.set_title("loss")
    ax.set_ylabel('loss')
    ax.set_xlabel('iterations')
    ax.legend()
    if save_plots:
        path = os.path.join(plot_folder, "LSTM_loss")
        fig.savefig(path)
    else:
        plt.show()

def plot_scores(scores_list, save_plots, plot_folder):
    fig, ax = plt.subplots()
    ax.plot(scores_list)
    ax.set_title("Score across epochs")
    ax.set_xlabel("iterations")
    ax.set_ylabel("score")
    if save_plots:
        path = os.path.join(plot_folder, "LSTM_score")
        fig.savefig(path)
    else:
        plt.show()