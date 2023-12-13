import os
import pandas as pd
import numpy as np
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from tqdm.notebook import tqdm

class SpectrogramDataset(Dataset):
    def __init__(self, data_path, file_ext, window_size, transform=None, time_step=5.12e-4, overlap_factor=0):
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

        # Precompute and store all windows with unique IDs using a dictionary
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
        window_dict = {key: value for key, value in found_dict.items()}
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
        data_shot = pd.read_pickle(file_path)

        return data_shot

    def get_start_end_longest_mode(self, label_sources, std_factor=0.25, mean_factor=0.8):
        """
        Compute the start and end times of the longest mode segment.

        Parameters:
        - label_sources (dict): Dictionary containing mode information.
        - std_factor (float): Standard deviation factor for adaptive thresholding.
        - mean_factor (float): Mean factor for adaptive thresholding.

        Returns:
        - start_longest_mode (float): Start time of the longest mode segment.
        - end_longest_mode (float): End time of the longest mode segment.
        """
        # Stack modes vertically and interpolate NaN values
        modes_stacked = np.vstack([label_sources[f"N{i}"] for i in range(5)])
        nan_indices = np.isnan(modes_stacked)
        modes_stacked[nan_indices] = np.interp(np.flatnonzero(nan_indices), np.flatnonzero(~nan_indices),
                                               modes_stacked[~nan_indices])

        # Normalize each mode to have a max value of 1
        modes_stacked /= np.max(modes_stacked, axis=1, keepdims=True)

        # Sum along axis 0
        modes = np.sum(modes_stacked, axis=0)

        # Applying Gaussian filter
        filtered_modes = gaussian_filter(modes, sigma=45)

        # Calculate adaptive threshold based on the filtered modes
        mode_thresh = self.compute_adaptive_threshold(filtered_modes, std_factor, mean_factor)

        # Find the longest mode segment above the threshold
        above_threshold = filtered_modes > mode_thresh
        change_points = np.diff(above_threshold).nonzero()[0] + 1
        if above_threshold[0]:
            change_points = np.insert(change_points, 0, 0)
        if above_threshold[-1]:
            change_points = np.append(change_points, len(above_threshold))

        segment_lengths = change_points[1::2] - change_points[::2]
        if len(segment_lengths) > 0:
            longest_segment_index = np.argmax(segment_lengths)
            start_longest_mode = label_sources["time"][change_points[2 * longest_segment_index]]
            end_longest_mode = label_sources["time"][change_points[2 * longest_segment_index + 1]]

            return start_longest_mode, end_longest_mode
        else:
            return None, None

    def compute_adaptive_threshold(self, modes, std_factor, mean_factor):
        """
        Compute adaptive threshold for mode segmentation.

        Parameters:
        - modes (np.ndarray): Mode values.
        - std_factor (float): Standard deviation factor for adaptive thresholding.
        - mean_factor (float): Mean factor for adaptive thresholding.

        Returns:
        - threshold (float): Adaptive threshold.
        """
        if np.all(np.isnan(modes)):
            return np.nan  # Return NaN if all values are NaN
        if np.std(modes) == 0:
            return np.nan  # Return NaN if standard deviation is zero
        mean_val = np.nanmean(modes)  # Use nanmean to ignore NaN values
        std_dev = np.nanstd(modes)  # Use nanstd to ignore NaN values

        return mean_factor * mean_val + std_factor * std_dev


    def compute_all_windows(self, overlap_factor=0):
        """
        Computes all windows with unique IDs for the dataset.

        Returns:
        - windows (list): A list of dictionaries, each containing information about a window.
        """
        windows = []
        unique_id = 0

        # For each experiment
        for shotno in tqdm(self.data_files, desc="Processing shots"):
            data_shot = self.load_shot(shotno)

            spec_odd = data_shot["x"]["spectrogram"]["OddN"].T

            frequency = data_shot["x"]["spectrogram"]["frequency"]
            time = data_shot["x"]["spectrogram"]["time"]

            # Compute sliding windows for OddN with overlap
            for i in range(0, len(time) - self.window_size + 1, self.window_step):
                start_idx = i
                end_idx = i + self.window_size

                slice_data = spec_odd[:, start_idx:end_idx]

                # Normalize and resize each slice
                if self.transform:
                    #print(f"slice_data.shape = {slice_data.shape}")
                    slice_data = self.transform(slice_data)
                    #print(f"slice_data.shape = {slice_data.shape}")

                # Calculate labels for each time stamp in the experiment
                start_longest_mode, end_longest_mode = self.get_start_end_longest_mode(data_shot['y']['modes'])
                labels = np.zeros_like(time)
                if start_longest_mode is not None and end_longest_mode is not None:
                    in_longest_mode = (time >= start_longest_mode) & (time <= end_longest_mode)
                    labels[in_longest_mode] = 1

                    # Check if 50% or more of the window's points have a label of 1
                    window_label = torch.tensor([1]) if np.mean(labels[start_idx:end_idx]) >= 0.5 else torch.tensor([0])

                    windows.append({
                        'unique_id': unique_id,
                        'window_odd': slice_data,  # The odd spectrogram
                        'frequency': frequency,
                        'time': time[start_idx:end_idx],
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'shotno': shotno,
                        'label': window_label
                    })

                    unique_id += 1

        # Print information about the generated windows
        total_windows = len(windows)
        print(f"The size of the odd spectrogram in the last element of windows: {windows[total_windows - 1]['window_odd'].shape}.")
        print(f"Total number of windows = {total_windows}")
        print(f"Number of unique IDs = {unique_id}")

        return windows

# Custom collate function
def custom_collate(batch):
    return batch
