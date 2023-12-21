import sys
sys.path.append('src/')

import argparse
import torch
import os
import matplotlib
import matplotlib.pyplot as plt

from data.LSTM_dataset import LSTMDataset, create_preds
from models.simple_LSTM import *
from models.metrics import *

font = {'size' : 18}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = (10, 8)
matplotlib.rcParams['figure.dpi'] = 150


def main(args) :

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device {device}')

    print(f'loading the model at {args.model_path}')
    model = torch.load(args.model_path).to(device)
    dataset = LSTMDataset(args.data_dir,
                            features_selection=args.features,
                            max_length=args.max_length,
                            without_label=True)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    print("predicting....")
    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)
    model.eval()
    with torch.no_grad():
        for (x_batch, metadata) in dataloader:

            shotno = metadata['shotno']
            x_batch = x_batch.to(device)

            output = model(x_batch)
            logits = torch.sigmoid(output).detach().cpu()
            preds = create_preds(logits)

            time = metadata['time']

            if args.batch_size is None :
                plot(shotno, logits, preds, time)
            else :
                for b_idx in range(len(shotno)):
                    plot(shotno[b_idx], logits[b_idx], preds[b_idx], time[b_idx])

    return 0

def plot(shotno, logits, preds, time):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"{shotno}")
    ax.plot(time, logits, label='logits', linestyle='--' )
    ax.plot(time, preds.squeeze(), label='output', linestyle='-')
    save_path = os.path.join(args.result_folder, f"{shotno}")
    ax.legend()
    fig.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path')
    parser.add_argument('data_dir')
    parser.add_argument('--max_length', help='maximum length of the sequence', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--result_folder", default="result")
    parser.add_argument('--features', nargs='+', default=['max_energies', 'N1'])

    args = parser.parse_args()
    exit(main(args))
