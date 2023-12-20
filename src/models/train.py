import sys
sys.path.append('../../src/')

import argparse
import torch
import torch.optim as optim
import numpy as np
import os

from data.LSTM_dataset import LSTMDataset, create_preds
from simple_LSTM import *
from early_stopping import EarlyStopping
from metrics import *
from train_LSTM_utils import train


def main(args) :
    if args.set_seed :
        torch.manual_seed(0)
        np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Creating the dataset")
    dataset = LSTMDataset(args.dataset_root,
                          max_length=args.max_length)
    
    size = int(len(dataset) * 0.9)

    train_set, test_set = torch.utils.data.random_split(dataset, [size, len(dataset)-size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size)
    print(f'train size : {len(train_set)}, val size : {len(test_set)}')

    pos_weight = compute_pos_weight(train_loader) 

    model = SimpleLSTM(args.input_size, 
                       args.hidden_size, 
                       args.num_layers, 
                       dropout_rate=args.dropout_rate)
    

    optimizer = optim.Adam(model.parameters(), 
                           lr=args.lr, 
                           weight_decay = args.weight_decay)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters')

    save_path = os.path.join(args.model_path, "lstm.pt")

    print("start the training")
    train(model, 
      train_loader, 
      val_loader, 
      optimizer, 
      criterion, 
      device, 
      compute_objective=comptue_score,
      n_epochs=args.n_epoch,
      l1_sigma=args.l1_sigma,
      direction="maximize",
      patience=args.atience,
      delta=args.delta,
      model_path=args.save_path)
    
    torch.save(model, save_path)

    print("Evaluation...")
    kappa = []

    for x_batch, y_batch in val_loader :
        x_batch = x_batch.to(device)
        
        logits = torch.sigmoid(model(x_batch)).detach().cpu()
        preds = create_preds(logits)

        kappa += compute_kappa_score(preds, y_batch)

    print(f'mean kappa : {np.mean(kappa)}')

    return 0

def comptue_score(logits, labels):
      logits = torch.sigmoid(logits).detach()
      preds = create_preds(logits)
      return compute_kappa_score(preds, labels)

def compute_pos_weight(data):
    size = 0
    nb_pos = 0
    for item in data:
        labels = item[1]
        size += len(labels.flatten())
        nb_pos +=(labels == 1).sum()
    print(f'size={size}, nb_pos={nb_pos}')
    return (size - nb_pos) / nb_pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_root' , help='root folder of the data, should contains either dataset_pickle or dataset_hickle and MHD_labels folder')
    parser.add_argument('hidden_size', type=int)
    parser.add_argument('num_layer', type=int)
    parser.add_argument('lr', type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--l1_sigma', default=0, type=float)
    parser.add_argument('--dropout_rate', default=0, type=float)
    parser.add_argument('--set_seed', action="store_true", type=int)
    parser.add_argument('--max_length', help='maximum length of the sequence', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--patience', default = 0, type=int)
    parser.add_argument('--early_stop_delta', default=1e-3, type=float)
    parser.add_argument('--model_path', default="models")
    parser.add_argument('--n_epoch', default=200, type=int)

    try :
        args = parser.parse_args()

        exit(main(args))
    except:
        parser.print_help()
        exit(0)

    



