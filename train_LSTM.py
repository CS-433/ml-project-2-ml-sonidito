import sys
sys.path.append('src/')

import argparse
import torch
import torch.optim as optim
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

from data.LSTM_dataset import LSTMDataset, create_preds
from models.simple_LSTM import *
from models.metrics import *
from models.train_LSTM_utils import train

font = {'size' : 18}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = (10, 8)
matplotlib.rcParams['figure.dpi'] = 150


def main(args) :

    g = None
    if args.set_seed :
        print("Seed fixed")
        torch.manual_seed(0)
        np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device {device}')

    print("Creating the dataset")
    dataset = LSTMDataset(args.dataset_root,
                          features_selection=args.features,
                          max_length=args.max_length)
    
    print(f'training with these featuers : {args.features}')
    
    train_ids = [86, 6, 43, 68, 27, 38, 7, 65, 29, 4, 87, 19, 53, 36, 51, 66, 59, 28, 37, 84, 77, 1, 16, 64, 30, 32, 62, 69, 17, 49, 79, 18, 24, 74, 57, 50, 56, 92, 11, 34, 73, 45, 54, 47, 89, 71, 82, 41, 76, 60, 48, 88, 2, 3, 80, 35, 46, 70, 90, 42, 20, 85, 9, 81, 21, 33, 75, 23, 83, 58, 5, 78, 52, 15, 31, 12, 63, 8, 93, 13, 14, 91, 61, 25]
    test_ids = [40, 22, 55, 72, 0, 26, 39, 67, 10, 44]

    train_set = torch.utils.data.Subset(dataset, train_ids)
    test_set =  torch.utils.data.Subset(dataset, test_ids)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size)
    print(f'train size : {len(train_set)}, val size : {len(test_set)}')
    print(f'batch_siez={args.batch_size}')

    pos_weight = compute_pos_weight(train_loader) 

    model = SimpleLSTM(dataset[0][0][0].shape[1], 
                       args.hidden_size, 
                       args.num_layers, 
                       dropout_rate=args.dropout_rate)
    
    print(model)
    
    optimizer = optim.Adam(model.parameters(), 
                           lr=args.lr, 
                           weight_decay = args.weight_decay)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters')

    save_path = os.path.join(args.model_path, "lstm.pt")
    if not os.path.exists(args.model_path) :
        os.makedirs(args.model_path)

    if args.save_plots and not os.path.exists(args.fig_folder) :
        os.makedirs(args.fig_folder)

    print(args.save_plots)

    print("start the training")
    model = train(model, 
      train_loader, 
      val_loader, 
      optimizer, 
      criterion, 
      device, 
      compute_objective=comptue_score,
      n_epochs=args.n_epoch,
      l1_sigma=args.l1_sigma,
      direction="maximize",
      patience=args.patience,
      delta=args.delta,
      model_path=save_path,
      save_plots=args.save_plots,
      plot_folder=args.fig_folder)
        
    torch.save(model, save_path)
    print(f"model saved at {save_path}")

    print("Evaluation...")
    kappa = []

    for (x_batch, shotno), y_batch in val_loader :
        x_batch = x_batch.to(device)
        
        logits = torch.sigmoid(model(x_batch)).detach().cpu()
        preds = create_preds(logits)

        kappa += compute_kappa_score(preds, y_batch)
        if args.save_eval :
            if args.batch_size is None :
                plot(shotno, logits, y_batch, preds, kappa)
            else :
                for b_idx in range(len(shotno)):
                    plot(shotno[b_idx], logits[b_idx], y_batch[b_idx], preds[b_idx], kappa[b_idx])

    print(f'mean kappa : {np.mean(kappa)}')
    if args.save_eval :
        print(f'prediction saved at {args.fig_folder}')

    return 0


def plot(shotno, logits, gt, preds, score):
    fig, ax = plt.subplots()
    ax.set_title(f"{shotno} - score {score:.3f}")
    ax.plot(logits, label='logits', linestyle='--' )
    ax.plot(gt, label='ground truth', alpha=0.8)
    ax.plot(preds.squeeze(), label='output', linestyle='-.')
    save_path = os.path.join(args.fig_folder, f"{shotno}")
    ax.legend()
    fig.savefig(save_path)
    plt.close()

def comptue_score(logits, labels):

      logits = torch.sigmoid(logits).detach()
      preds = create_preds(logits)
      return compute_kappa_score(preds, labels)

def compute_pos_weight(data):
    size = 0
    nb_pos = 0
    for _, labels in data:
        size += len(labels.flatten())
        nb_pos +=(labels == 1).sum()
    print(f'size={size}, nb_pos={nb_pos}')
    return (size - nb_pos) / nb_pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_root' , help='root folder of the data, should contains either dataset_pickle or dataset_hickle and MHD_labels folder')
    parser.add_argument('hidden_size', type=int)
    parser.add_argument('num_layers', type=int)
    parser.add_argument('lr', type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--l1_sigma', default=0, type=float)
    parser.add_argument('--dropout_rate', default=0, type=float)
    parser.add_argument('--set_seed', action="store_false")
    parser.add_argument('--max_length', help='maximum length of the sequence', type=int, default=4096)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--patience', default = 0, type=int)
    parser.add_argument('--delta', default = 1e-3, type=float)
    parser.add_argument('--model_path', default="models")
    parser.add_argument('--n_epoch', default=200, type=int)
    parser.add_argument('--save_plots', action="store_false")
    parser.add_argument('--fig_folder', default="figs")
    parser.add_argument('--save_eval', action="store_false")
    parser.add_argument('--features', nargs='+', default=['max_energies', 'N1'])

    # try :
    #     args = parser.parse_args()

    #     exit(main(args))
    # except:
    #     parser.print_help()
    #     exit(0)

    args = parser.parse_args()
    exit(main(args))


    



