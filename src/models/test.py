import sys
sys.path.append('../../src/')

import argparse
import torch
import torch.optim as optim
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

from data.LSTM_dataset import LSTMDataset, create_preds
from simple_LSTM import *
from metrics import *
from train_LSTM_utils import train


def main(args) :

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f'loading the model at {args.model_path}')
    model = torch.load(args.model_path).to(device)
    dataset = LSTMDataset(args.data_dir,
                            max_length=args.max_length)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    print("predicting....")
    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)
    model.eval()
    with torch.no_grad():
        for (x_batch, shotno), y_batch in dataloader:
            x_batch = x_batch.to(device)

            output = model(x_batch)
            logits = torch.sigmoid(output).detach().cpu()
            preds = create_preds(logits)

            if args.batch_size is None :
                plot(shotno, logits, y_batch.cpu(), preds)
            else :
                for b_idx in range(len(shotno)):
                    plot(shotno[b_idx], logits[b_idx], y_batch[b_idx], preds[b_idx])

    return 0

def plot(shotno, logits, gt, preds):
    fig, ax = plt.subplots()
    ax.set_title(shotno)
    ax.plot(logits, label='logits', linestyle='--' )
    ax.plot(gt, label='ground truth', alpha=0.8)
    ax.plot(preds.squeeze(), label='output', linestyle=':')
    save_path = os.path.join(args.result_folder, f"{shotno}")
    fig.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path')
    parser.add_argument('data_dir')
    parser.add_argument('--max_length', help='maximum length of the sequence', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--result_folder", default="result")

    args = parser.parse_args()
    exit(main(args))
