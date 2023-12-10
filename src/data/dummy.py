import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import glob
import pickle
import os
import numpy as np
from scipy.ndimage import gaussian_filter


import matplotlib.pyplot as plt

class DummyDataset(Dataset):
    def __init__(self, data_path,  transform=None, max_length = None):
        self.data_path = data_path
        self.all_shots = [int(os.path.basename(x.split(f".pickle")[0])) 
             for x in glob.glob(os.path.join(data_path, f"*.pickle"))]
        
        self.transform = transform
        self.max_length = max_length
        

    def __len__(self):
        return len(self.all_shots)

    def __getitem__(self, idx):
        shotno = self.all_shots[idx]
        data = self.load_shot(shotno)
        spec_odd =  data['x']['spectrogram']['OddN']

        if self.transform != None:
            spec_odd = self.transform(spec_odd)

        mask = ~torch.isinf(spec_odd) & ~torch.isnan(spec_odd)

        spec_odd = spec_odd + spec_odd[mask].min()

        std, mean = torch.std_mean(spec_odd[mask])
        spec_odd[~mask] = mean

        max_energies = self.compute_max_energies((spec_odd - mean) / std)

        labels = self.compute_labels(max_energies)

        mode_n1 = torch.from_numpy(data['y']['modes']['N1'])
        mode_n1[torch.isnan(mode_n1)] = 0

        if self.max_length != None:
            if len(labels) > self.max_length:
                labels = labels[:self.max_length]
                max_energies = max_energies[:self.max_length]
            elif len(labels) < self.max_length:
                temp = torch.zeros(self.max_length)
                temp[:max_energies.size(0)] = max_energies
                max_energies = temp

                temp = torch.zeros(self.max_length)
                temp[:labels.size(0)] = labels
                labels = temp


        mode_n1 = self.normalize_mode(mode_n1)
        mode_n1 = F.interpolate(mode_n1.unsqueeze(0).unsqueeze(0), size=len(max_energies)).squeeze()
        # mode_n1 = torch.from_numpy(gaussian_filter(mode_n1, sigma=10))

        input = torch.vstack((max_energies.float(), mode_n1.float()))
        input = input.swapaxes(0,1)

        # return torch.from_numpy(gaussian_filter(torch.mean(spec_odd, dim=-1),10)).float().unsqueeze(-1), labels
        # return spec_odd, labels
        return input, labels
        # return max_energies.float().unsqueeze(-1), labels

    def load_shot(self, shotno):
        with open(os.path.join(self.data_path, f"{shotno}.pickle"), "rb") as f:
            return pickle.load(f)

    def compute_max_energies(self, spec):
        max_values, _ = torch.max(torch.from_numpy(gaussian_filter(spec, 10)).squeeze(), dim = 1)
        max = torch.max(max_values)
        max_values = torch.from_numpy(gaussian_filter(max_values / max, 10)).squeeze()

        return max_values
    
    def compute_labels(self, max_energies, threshold=0.8):
        start = np.argmax(max_energies > threshold)
        end = len(max_energies) - torch.argmax(torch.flip(max_energies > threshold, dims=(0,)) * 1)
 
        labels = torch.zeros(len(max_energies))
        labels[start:end] = 1

        return labels

    def normalize_mode(self, mode_val):
        std, mean = torch.std_mean(mode_val)

        return (mode_val - mean) / std
    

        