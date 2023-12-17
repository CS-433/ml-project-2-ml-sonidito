import torch
from tqdm.autonotebook import tqdm
from torch.utils.data import Dataset
import torch.fft as fft
import torch.nn.functional as F
import glob
import pickle
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from itertools import groupby



import matplotlib.pyplot as plt

class LSTMDataset(Dataset):
    def __init__(self, data_path, labels_path,  features_selection = None, transform=None, max_length = None):
        """
        param:
            - features_selection : select which features to used for the training, 0 is the max_energy 1 to 6 is the mode
        """
        
        self.data_path = data_path
        self.labels_path = labels_path
        self.all_shots = [int(os.path.basename(x.split(f".pickle")[0])) 
             for x in glob.glob(os.path.join(data_path, f"*.pickle"))]
        
        self.transform = transform
        self.max_length = max_length
 
        self.features_selection = features_selection

        self.inputs = []
        self.labels = []

        with tqdm(self.all_shots) as pbar:
            pbar.set_description('data processing')
            for shotno in pbar :
                data_features = self.load_shot(shotno)
                data_label = self.load_label(shotno)

                features = self.process_data(data_features, data_label['time'][0])
                label = self.process_labels(shotno, torch.tensor(data_features['x']['spectrogram']['time']))

                self.inputs.append(features)
                self.labels.append(label)


    def __len__(self):
        return len(self.all_shots)

    def __getitem__(self, idx):
        if self.features_selection is not None:
            input = [self.inputs[idx][key].numpy() for key in self.features_selection]
        else:
            input = [self.inputs[idx][key].numpy() for key in self.inputs[idx].keys()]

        return torch.tensor(input, dtype=torch.float32).swapaxes(0,1), self.labels[idx]

    def load_shot(self, shotno):
        with open(os.path.join(self.data_path, f"{shotno}.pickle"), "rb") as f:
            return pickle.load(f)
        
    def load_label(self, shotno):
        with open(os.path.join(self.labels_path, f'TCV_{shotno}_apau_MHD_labeled.csv')) as f:
            data = np.loadtxt(f, skiprows=1, delimiter=',')
            return {'time' : data[:, 0],
                    'label' : data[:, 4]}
    
    # def compute_labels(self, max_energies, threshold=0.8):
    #     """
    #         Create pseudo label by thresholding the energy
    #     """
    #     start = np.argmax(max_energies > threshold)
    #     end = len(max_energies) - torch.argmax(torch.flip(max_energies > threshold, dims=(0,)) * 1)
 
    #     labels = torch.zeros(len(max_energies))
    #     labels[start:end] = 1

    #     return labels
    
    def compute_max_energy(self, spec_odd) : 
        """
            Search the max energy of the spectogram by selecting the max value of each timestamp
        """

        mask = ~torch.isinf(spec_odd) & ~torch.isnan(spec_odd)

        spec_odd = spec_odd + spec_odd[mask].min()

        std, mean = torch.std_mean(spec_odd[mask])
        spec_odd[~mask] = mean

        spec_odd = (spec_odd - mean) / std

        # filtered = torch.from_numpy(gaussian_filter(spec_odd, 10)).squeeze()
        # max_energies, _ = torch.max(filtered, dim = 1)
        # freq_position = torch.argmax(filtered, dim=1)
        # max = torch.max(max_energies)
        # max_energies = torch.from_numpy(gaussian_filter(max_energies / max, 10)).squeeze()

        max_energies, _ = torch.max(spec_odd, dim = 1)
        freq_position = torch.argmax(spec_odd, dim=1)
        max = torch.max(max_energies)
        max_energies = max_energies / max
        
        return torch.from_numpy(gaussian_filter(max_energies, 1)), freq_position
    
    def compute_mode(self, mode, size) : 
        """
            Process the mode value, normalize and scale the value to have value between [0,1]
        """        
        mode = self.normalize_mode(mode)
        mode = mode - mode.min()
        mode = mode / mode.max()
        mode = F.interpolate(mode.unsqueeze(0).unsqueeze(0), size=size).squeeze()

        return mode

    def normalize_mode(self, mode_val):
        std, mean = torch.std_mean(mode_val)

        return (mode_val - mean) / std
    
    def process_data(self, data, label_start) : 
        """
            Process the data

            return :
                - input : data used for the training
                - label : (#len(data), *) list of bool indicating if the corresponding timestamp has a perturbation
        """

        spec_odd =  data['x']['spectrogram']['OddN']
        spec_time = data['x']['spectrogram']['time']

        spec_odd = spec_odd[spec_time >= label_start, :] # Keep only those that have a label

        if self.transform != None:
            spec_odd = self.transform(spec_odd)

        max_energies, position = self.compute_max_energy(spec_odd)

        input = {}

        input['max_energies'] = max_energies
        input['position'] = position

        # labels = self.process_labels()

        modes = data['y']['modes']

        for i in range(5):
            mode = torch.from_numpy(modes[f'N{i}'])
            mode[torch.isnan(mode)] = 0
            mode = self.compute_mode(mode, len(max_energies))

            input[f'N{i}'] = mode

        mode = torch.from_numpy(modes['LM'])
        mode[torch.isnan(mode)] = 0
        mode = self.compute_mode(mode, len(max_energies))

        input['LM'] = mode

        if self.max_length != None:
            if len(input['max_energies']) > self.max_length:
                for k, v in input.items():
                    input[k] = v[:self.max_length]
            elif len(input) < self.max_length:
                for k, v in input.items():
                    temp = torch.zeros(self.max_length)
                    temp[:v.size(0)] = v
                    input[k] = temp

                # temp = torch.zeros(self.max_length)
                # temp[:labels.size(0)] = labels
                # labels = temp

        return input #, labels
    
    def process_labels(self, shotno, spec_time):

        label_time = []
        labels = []
        with open(os.path.join(self.labels_path, f'TCV_{shotno}_apau_MHD_labeled.csv')) as f:
            data = np.loadtxt(f, skiprows=1, delimiter=',')

            label_time = data[:, 0]
            labels = data[:, 4]

        perturbation_idx = np.where(labels == 2)[0]
        true_labels = torch.zeros(len(spec_time))

        for _, g in groupby(enumerate(perturbation_idx), lambda k: k[0] - k[1]):
            start = next(g)[1]
            end = list(v for _, v in g)[-1] or [start]
            
            start_time = label_time[start]
            end_time = label_time[end]

            idx = torch.where((spec_time >=  start_time) & (spec_time <= end_time))
            true_labels[idx] = 1


        if len(true_labels) > self.max_length:
            true_labels = true_labels[:self.max_length]
        elif len(true_labels) < self.max_length:
            temp = torch.zeros(self.max_length)
            temp[:true_labels.shape[0]] = true_labels
            true_labels = temp

        return true_labels
    
    def compute_pos_weight(self):
        size = 0
        nb_pos = 0
        for l in self.labels:
            size += len(l)
            nb_pos += torch.sum(l == 1)
        return size / nb_pos

