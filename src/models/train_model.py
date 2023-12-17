from tqdm.autonotebook import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from models.metrics import *
from models.early_stopping import EarlyStopping

from models.simple_LSTM import SimpleLSTM
import torch.optim as optim
import torch.nn as nn


def train(model,train_loader, val_loader, optimizer, criterion, device,
            n_epochs = 10, early_stopping = None, l1_sigma=0, compute_score = None ):

    model.to(device)

    train_losses = []
    val_losses = []

    scores_list = []

    eval_loss, scores = run_validation(model, val_loader, criterion, device, compute_score)
    val_losses.append(eval_loss)
    if compute_score is not None:
        scores_list.append(scores.item())

    with tqdm(range(n_epochs)) as pbar: 
        for epoch in pbar:
            epoch_losses = train_one_epoch(model, train_loader, criterion, optimizer, epoch, l1_sigma, device, leave=False)
            train_losses += epoch_losses

            eval_loss, scores = run_validation(model, val_loader, criterion, device, compute_score)
            val_losses.append(eval_loss)
            if compute_score is not None :
                scores_list.append(scores.item())
            else :
                scores = eval_loss
            pbar.set_postfix(score=scores.item(), train_loss=np.mean(train_losses))

            if early_stopping != None:
                if early_stopping.early_stop(scores, model):
                    model = torch.load(early_stopping.save_path) # laod best model
                    break
    
    plot_losses(train_losses, val_losses)

    plt.plot(-1 * np.array(scores_list))
    plt.show()


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, l1_sigma, device, disable_progress_bar=False, leave=True):
    model.train()
    batch_losses = []
    with tqdm(train_loader, unit='batch', disable=disable_progress_bar, leave=leave) as tepoch:
        for x_batch, y_batch in tepoch :
            tepoch.set_description(f'Epoch {epoch+1}')

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            y_batch = y_batch.to(torch.float32)

            loss = criterion(y_pred, y_batch)
            if l1_sigma != 0 :
                for param in model.parameters():
                    if param.requires_grad:
                        loss = loss + l1_sigma * torch.norm(param, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            tepoch.set_postfix(loss=np.mean(batch_losses))
    # print(f'Epoch {epoch + 1} loss : {np.mean(batch_losses)}')
    return batch_losses

def run_validation(model, val_loader, criterion, device, compute_score = None) :
    model.eval()    
    running_val_loss = 0
    scores = torch.tensor([]).to(device)
    with torch.no_grad():
        for x_batch, y_batch in val_loader :

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            y_batch = y_batch.to(torch.float32)
            loss = criterion(logits, y_batch)

            if compute_score is not None:
                scores = torch.cat((compute_score(logits, y_batch), scores))

            running_val_loss += loss.item()
    return running_val_loss / len(val_loader), torch.mean(scores)

def k_fold(dataset, model, criterion, device, lr, weight_decay,
           k_fold=5, n_epochs = 0, l1_sigma=0, batch_size=1, patience=5, delta=5e-2, threshold=0.7, compute_score= None, disable_pbar=False) :
    
    def create_output(logits, threshold=0.8):
        output = torch.zeros(logits.shape)
        for idx, logit in enumerate(logits):
            if (logit < threshold).all():
                continue

            start = torch.argmax((logit > threshold)* 1)
            end = len(logit) - torch.argmax(torch.flip(logit > threshold, dims=(0,)) * 1)

            res = torch.zeros(len(logit))
            res[start:end] = 1
            output[idx] = res

        return output
    
    model.to(device)

    kd = KFold(k_fold, shuffle=True)

    mean_accuracy = 0
    mean_precision = 0
    mean_recall = 0
    mean_f1 = 0
    mean_train_loss = 0
    mean_eval_loss = 0
    mean_kappa = 0
    

    for fold, (train_ids, test_ids) in tqdm(enumerate(kd.split(dataset)), disable=disable_pbar, total=k_fold):
        train_subsample = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsample = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_subsample)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_subsample)

        early_stopping = None
        if patience is not None:
            early_stopping = EarlyStopping('../models/k_fold_lstm.pt', patience, delta)
    
        model.reset_weights()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            epoch_losses = train_one_epoch(model, train_loader, criterion, optimizer, epoch, l1_sigma, device, disable_progress_bar=True)
            train_losses += epoch_losses

            eval_loss, scores = run_validation(model, val_loader, criterion, device, compute_score)
            val_losses.append(eval_loss)

            if compute_score is None:
                scores = eval_loss

            if early_stopping is not None and early_stopping.early_stop(scores, model):
                model = torch.load(early_stopping.save_path) # laod best model
                break
        
        # plot_losses(train_losses, val_losses)

        mean_train_loss += np.mean(train_losses) / k_fold
        mean_eval_loss += np.mean(val_losses) / k_fold

        model.eval()
        with torch.no_grad():
            total_accuracy = 0
            total_precision = 0
            total_recall = 0
            total_f1 = 0
            total_kappa = 0
            for x_batch, y_batch in val_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch)
                logits = torch.sigmoid(logits)

                preds = create_output(logits, threshold).to(device)

                # for idx, pred in enumerate(preds[:2]):
                #     plt.plot(logits[idx].cpu(), label='logits', linestyle='--' )
                #     plt.plot(y_batch[idx].cpu(), label='ground truth', alpha=0.8)
                #     plt.plot(pred.cpu().squeeze(), label='output', linestyle=':')
                #     plt.legend()
                #     plt.show()

                total_accuracy += torch.sum(compute_accuracy(preds, y_batch)) 
                total_precision += torch.sum(compute_precision(preds, y_batch))
                total_recall += torch.sum(compute_recall(preds, y_batch)) 
                total_f1 += torch.sum(compute_f1(preds, y_batch))
                total_kappa += torch.sum(compute_kappa_score(preds, y_batch))

            mean_accuracy += total_accuracy / len(val_loader.dataset) / k_fold
            mean_precision += total_precision / len(val_loader.dataset) / k_fold
            mean_recall += total_recall / len(val_loader.dataset) / k_fold
            mean_f1 += total_f1 / len(val_loader.dataset) / k_fold       
            mean_kappa += total_kappa / len(val_loader.dataset) /k_fold       
    
    return {'accuracy': mean_accuracy.item(),
            'precision' : mean_precision.item(),
            'recall' : mean_recall.item(),
            'f1' : mean_f1.item(),
            'kappa' : mean_kappa.item(),
            'train_loss' : mean_train_loss,
            'val_loss' : mean_eval_loss}


def plot_losses(train_losses, val_losses):
    plt.plot(np.linspace(0, len(val_losses)-1, len(train_losses)), train_losses, label='train')
    plt.plot(val_losses, label='validation')
    plt.title("loss")
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.legend()
    plt.show()