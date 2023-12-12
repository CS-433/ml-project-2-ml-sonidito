import os
import pandas as pd
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, transforms
import matplotlib.pyplot as plt

class SpectrogramDataset(Dataset):
    def __init__(self, data_path, file_ext, window_size, transform=None, time_step=5.12e-4, overlap_factor = 0):
        """
        SpectrogramDataset constructor.

        Parameters:
        - data_path (str): Location of the data.
        - file_ext (str): File type of the data.
        - window_size (int): Number of time steps in each window.
        - transform (callable): Optional transform to be applied on a window.
        - time_step (float): Duration of each time step in ms.
        - overlap_factor (float): Fraction of overlap between consecutive windows (0 to 1).

        Attributes:
        - data_path (str): Location of the data.
        - file_ext (str): File type of the data.
        - window_size (int): Number of time steps in each window.
        - transform (callable): Optional transform to be applied on a window.
        - time_step (float): Duration of each time step in ms.
        - overlap_factor (float): Fraction of overlap between consecutive windows (0 to 1).
        - window_step (int): Step size for moving the window.

        Initializes the dataset by obtaining all shot numbers and precomputing all windows with unique IDs.
        """
        self.data_path = data_path
        self.file_ext = file_ext
        self.window_size = window_size
        self.transform = transform
        self.time_step = time_step
        self.overlap_factor = overlap_factor
        self.window_step = int(self.window_size * (1 - overlap_factor))  # Step size for moving the window

        # Obtain all shot numbers
        self.data_files = [int(os.path.basename(x.split(f".{file_ext}")[0]))
                           for x in glob.glob(os.path.join(data_path, f"*.{file_ext}"))]

        # Precompute and store all windows with unique IDs using a dictionnary
        self.windows = self.compute_all_windows()

    def __len__(self):
        """
        Returns the total number of windows in the dataset.
        """
        return len(self.windows)
    

    def __getitem__(self, idx):
        """
        Returns a single window based on the provided idx (unique identifier).

        Parameters:
        - idx (int): Unique identifier for the window.

        Returns:
        - window_dict (dict): A dictionary containing information about the window.
        """
        found_dict = next((my_dict for my_dict in self.windows if my_dict.get('unique_id') == idx), None)
        window_dict = {key: [value] for key, value in found_dict.items()}
        return window_dict

    def load_shot(self, shotno):
        """
        Loads data for a specific experiment.

        Parameters:
        - shotno (int): Experiment (shot) number.

        Returns:
        - data_shot (pd.DataFrame): Data for the specified experiment.
        """
        file_path = os.path.join(self.data_path, f"{shotno}.{self.file_ext}")
        return pd.read_pickle(file_path)
    
    def compute_all_windows(self, overlap_factor=0):
        """
        Computes all windows with unique IDs for the dataset.

        Returns:
        - windows (list): A list of dictionaries, each containing information about a window.
        """
        windows = []
        unique_id = 0

        # For each experiment
        for shotno in self.data_files:
            data_shot = self.load_shot(shotno)

            spec_odd = torch.tensor(data_shot["x"]["spectrogram"]["OddN"], dtype=torch.float32).T
        
            frequency = data_shot["x"]["spectrogram"]["frequency"]
            time = data_shot["x"]["spectrogram"]["time"]

            # Compute sliding windows for OddN with overlap
            for i in range(0, len(time) - self.window_size + 1, self.window_step):
                start_idx = i
                end_idx = i + self.window_size

                slice_data = spec_odd[:, start_idx:end_idx]

                windows.append({
                    'unique_id': unique_id,
                    'window_odd': slice_data, # The odd spectrogram
                    'frequency': frequency,
                    'time': time[start_idx:end_idx],
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'shotno': shotno
                })
                

                unique_id += 1

        # Print information about the generated windows
        total_windows = len(windows)
        print(f"The size of the odd spectrogram in the last element of windows: {windows[total_windows-1]['window_odd'].shape}.")
        print(f"Total number of windows = {total_windows}")
        print(f"Number of unique IDs = {unique_id}")

        return windows
    
# Custom collate function
def custom_collate(batch):
    return batch    


