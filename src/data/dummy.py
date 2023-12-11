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

        self.inputs = []
        self.labels = []

        for shotno in self.all_shots :
            data = self.load_shot(shotno)
            input, label = self.process_data(data)

            self.inputs.append(input)
            self.labels.append(label)


    def __len__(self):
        return len(self.all_shots)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

    def load_shot(self, shotno):
        with open(os.path.join(self.data_path, f"{shotno}.pickle"), "rb") as f:
            return pickle.load(f)
    
    def compute_labels(self, max_energies, threshold=0.8):
        """
            Create pseudo label by thresholding the energy
        """
        start = np.argmax(max_energies > threshold)
        end = len(max_energies) - torch.argmax(torch.flip(max_energies > threshold, dims=(0,)) * 1)
 
        labels = torch.zeros(len(max_energies))
        labels[start:end] = 1

        return labels
    
    def compute_max_energy(self, spec_odd) : 
        """
            Search the max energy of the spectogram by selecting the max value of each timestamp
        """

        if self.transform != None:
            spec_odd = self.transform(spec_odd)

        mask = ~torch.isinf(spec_odd) & ~torch.isnan(spec_odd)

        spec_odd = spec_odd + spec_odd[mask].min()

        std, mean = torch.std_mean(spec_odd[mask])
        spec_odd[~mask] = mean

        spec_odd = (spec_odd - mean) / std

        max_energies, _ = torch.max(torch.from_numpy(gaussian_filter(spec_odd, 10)).squeeze(), dim = 1)
        max = torch.max(max_energies)
        max_energies = torch.from_numpy(gaussian_filter(max_energies / max, 10)).squeeze()

        return max_energies
    
    def compute_mode(self, mode, size) : 
        """
            Process the mode value, normalize and scale the value to have value between [0,1]
        """        
        mode = self.normalize_mode(mode)
        mode = mode / mode.max()
        mode = F.interpolate(mode.unsqueeze(0).unsqueeze(0), size=size).squeeze()

        return mode

    def normalize_mode(self, mode_val):
        std, mean = torch.std_mean(mode_val)

        return (mode_val - mean) / std
    
    def process_data(self, data) : 
        """
            Process the data

            return :
                - input : data used for the training
                - label : (#len(data), *) list of bool indicating if the corresponding timestamp has a perturbation
        """

        spec_odd =  data['x']['spectrogram']['OddN']
        max_energies = self.compute_max_energy(spec_odd)

        labels = self.compute_labels(max_energies)

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

        mode_n1 = torch.from_numpy(data['y']['modes']['N1'])
        mode_n1[torch.isnan(mode_n1)] = 0

        mode_n1 = self.compute_mode(mode_n1, len(max_energies))

        input = torch.vstack((max_energies.float(), mode_n1.float()))
        input = input.swapaxes(0,1)
        
        return input, labels