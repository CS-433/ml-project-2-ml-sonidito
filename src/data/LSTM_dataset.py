import torch
from tqdm.autonotebook import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F
import glob
import pickle
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from itertools import groupby
from sklearn.preprocessing import MinMaxScaler

class LSTMDataset(Dataset):
    """
        Dataset used to train LSTM model

        Attributes
            all_shots : list
                List containing all shot number
            transform : torchvision.transform
                Transformation to apply to the spectrogram
            max_length : int
                Max length of the data
            features_selection : list
                Select which features to used for the training, valid value : max_energies, N0, N1, N2, N3, N4, LM
            features : list
                processed features that can be used for the training
            label : list 
                List of labels
            max_energy_mean : float
                Mean of the max_energy across all training set
            max_energy_std : float
                Std of the max_energy across all training set
    """
    def __init__(self, data_root, features_selection = None, transform=None, max_length = None, use_pickle = True):
        """
        Parameters
            data_root : str
                Path to the folder containing the pickle or hickle folder and the labels
            features_selection : list, optional
                Select which features to used for the training, valid value : max_energies, N0, N1, N2, N3, N4, LM
            transform : torchvision.transform, optional
                Transformation to apply to the spectrogram
            max_length : int, optional
                Max length of the data
            use_pickle : bool, optional
                Use pickle instead of hickle to load data
        """

        if use_pickle:
            import pickle
            data_path = os.path.join(data_root, "dataset_pickle")
            file_ext = "pickle"
        else:
            import hickle
            data_path = os.path.join(data_root,"dataset_h5")
            file_ext = "h5"

        labels_path = os.path.join(data_root, "MHD_labels")
        
        self.all_shots = [int(os.path.basename(x.split(f".{file_ext}")[0])) 
             for x in glob.glob(os.path.join(data_path, f"*.{file_ext}"))]
        
        self.transform = transform
        self.max_length = max_length
 
        self.features_selection = features_selection

        self.features = []
        self.labels = []

        self.scaler = MinMaxScaler()

        # Standardization of the energy
        self.min = {'max_energies' : -85.966,
                    'N0' : 0,
                    'N1' : 0,
                    'N2' : 0,
                    'N3' : 0,
                    'N4' : 0,
                    'LM' : 0}
        
        self.max = {'max_energies' : 0,
            'N0' : 0.0151,
            'N1' : 0.0142,
            'N2' : 0.006,
            'N3' : 0.006,
            'N4' : 0.004,
            'LM' : 0.001}
        
        with tqdm(self.all_shots) as pbar:
            pbar.set_description('data processing')
            for shotno in pbar :
                data_features = self.load_shot(shotno, data_path, file_ext, use_pickle)
                data_label = self.load_label(shotno, labels_path)

                features = self.process_features(data_features, data_label['time'][0])
                label = self.process_labels(data_label, torch.tensor(data_features['x']['spectrogram']['time']))

                self.features.append(features)
                self.labels.append(label)

            self.remove_empty_mode_features()
            self.padding()

    def __len__(self):
        return len(self.all_shots)

    def __getitem__(self, idx):
        if self.features_selection is not None:
            input = [self.features[idx][key].numpy() for key in self.features_selection]
        else:
            input = [self.features[idx][key].numpy() for key in self.features[idx].keys()]

        return torch.tensor(input, dtype=torch.float32).swapaxes(0,1), self.labels[idx]

    def load_shot(self, shotno, data_path, file_ext, use_pickle):
        with open(os.path.join(data_path, f"{shotno}.{file_ext}"), "rb") as f:
            if use_pickle:
                return pickle.load(f)
            else:
                return hickle.load(f)
        
    def load_label(self, shotno, labels_path):
        """
        Extract the time and the label from the label csv

        Parameters
            shotno : int
                Experience number
            label_path : str
                path to the folder containing labels

        Return 
            dict 
                A dictionary containing the time and the label
        """
        with open(os.path.join(labels_path, f'TCV_{shotno}_apau_MHD_labeled.csv')) as f:
            data = np.loadtxt(f, skiprows=1, delimiter=',')
            return {'time' : data[:, 0],
                    'label' : data[:, 4]}
    
    def compute_max_energy(self, spec, f, freq_end = 800) : 
        """
        Search the max energy of the spectrogram by selecting the max value of each timestamp

        Parameters :
            spec : torch.tensor (W,H)
                The spectrogram in tensor form 

        """
        mask = ~torch.isinf(spec) & ~torch.isnan(spec)

        _, mean = torch.std_mean(spec[mask])
        spec[~mask] = mean
        max_energies,_ = torch.max(spec, dim = 1)

        max_energies = torch.from_numpy(gaussian_filter(max_energies, 3))

        max_energies = self.normalize('max_energies', max_energies)
        return max_energies
    
    def compute_mode(self, mode, size) : 
        """
            Process the mode value, use a nearest interpolation to downscale the mode and compute a max-min scaling.
            If the mode contain nan or inf value, replace it with the mean
        
        Parameters
            mode : torch.tensor
                Tensor containing the mode values
            size : int
                The length of the mode
        
        Return 
            torch.tensor
                Tensor of mode interpolated and scaled
        """        
        mask = ~torch.isinf(mode) & ~torch.isnan(mode)
        _, mean = torch.std_mean(mode[mask])

        mode[~mask] = 0
        mode = F.interpolate(mode.unsqueeze(0).unsqueeze(0), size=size).squeeze()

        return mode

    # def standardize_mode(self, mode_val):
    #     std, mean = torch.std_mean(mode_val)

    #     return (mode_val - mean) / std
    
    def process_features(self, features, label_start) : 
        """
            Process the features, remove the timestamp that precedes to the label start 

            Parameter
                features : dict
                    Dictionary containing the feature to process
                label start : float
                    The start timestamp of the label

            Return :
                dict
                    Dictionary containing the processed features
                    List of features: max_energies, N0, N1, N2, N3, N4m LM
        """

        spec_odd =  torch.from_numpy(features['x']['spectrogram']['OddN'])
        spec_time = features['x']['spectrogram']['time']
        f =  features['x']['spectrogram']['frequency'] 

        spec_odd = spec_odd[spec_time >= label_start, :] # Keep only those that have a label

        if self.transform != None:
            spec_odd = self.transform(spec_odd)

        # max_energies, position = self.compute_max_energy(spec_odd, f)
        max_energies = self.compute_max_energy(spec_odd, f)

        input = {}

        input['max_energies'] = max_energies
        # input['position'] = position

        modes = features['y']['modes']
        for i in range(5):
            mode = torch.from_numpy(modes[f'N{i}'])
            mode[torch.isnan(mode)] = 0
            mode = self.compute_mode(mode, len(max_energies))
            mode = self.normalize(f'N{i}', mode)

            input[f'N{i}'] = mode

        mode = torch.from_numpy(modes['LM'])
        mode[torch.isnan(mode)] = 0
        mode = self.compute_mode(mode, len(max_energies))
        mode = self.normalize('LM', mode)

        input['LM'] = mode

        return input
    
    def normalize(self, key, data) :
        return (data - self.min[key]) / (self.max[key] - self.min[key])
    
    def padding(self) : 
        if self.max_length != None:
            for idx, data in enumerate(self.features) : 
                if len(data['max_energies']) > self.max_length:
                    for k, v in data.items():
                        self.features[idx][k] = v[:self.max_length]
                elif len(data) < self.max_length:
                    for k, v in data.items():
                        temp = torch.zeros(self.max_length)
                        temp[:v.size(0)] = v
                        self.features[idx][k] = temp
        
    
    def process_labels(self, data, spec_time):
        """
        Create the label according to the spectrogram timestamp

        Parameters 
            data : list
                label data, 2 equals perturbation, 1 otherwise
            spec_time : list
                Timestamp of the spectrogram

        Return 
            torch.tensor
                List of the label with the same timestamp than the spectrogram
            
        """

        label_time = data['time']
        labels = data['label']

        perturbation_idx = np.where(labels == 2)[0] # Retrieve idx where a perturbation occurred
        true_labels = torch.zeros(len(spec_time))

        for _, g in groupby(enumerate(perturbation_idx), lambda k: k[0] - k[1]): # Iterate by positive block
            start = next(g)[1]
            end = list(v for _, v in g)[-1] or [start]
           
            start_time = label_time[start]
            end_time = label_time[end]

            if end_time - start_time <= 0.03: # Skip small mode
                continue

            # Search the idx in the spectrogram where it contains the perturbation
            idx = torch.where((spec_time >=  start_time) & (spec_time <= end_time)) 
            true_labels[idx] = 1

        # Cut or pad if necessary 
        if self.max_length is not None:
            if len(true_labels) > self.max_length:
                true_labels = true_labels[:self.max_length]
            elif len(true_labels) < self.max_length:
                temp = torch.zeros(self.max_length)
                temp[:true_labels.shape[0]] = true_labels
                true_labels = temp

        return true_labels

    def remove_empty_mode_features(self):
        """
            Remove from the data that contains only negative label
        """
        indexes = np.where((np.array(self.labels) == 0).all(axis = 1))[0]
        
        if len(indexes) > 0 :
            print(f"deleted {len(indexes)} data that doesn't have any mode")
            self.labels = [elem for idx, elem in enumerate(self.labels) if idx not in indexes]
            self.features = [elem for idx, elem in enumerate(self.features) if idx not in indexes]
            self.all_shots = [elem for idx, elem in enumerate(self.all_shots) if idx not in indexes]

            assert len(self.labels) == len(self.features)

    
def create_preds(logits, threshold=0.8):
    """
        For the given logits, use a threshold to create the prediction

        Parameters
            logits : torch.tensor
                logits than will be transformed into predictions
            threshold : float, optional
                logits greater that the threshold will be assigned the value 1, 0 otherwise

        Return
            torch.tensor 
                Predictions 
    """
    output = torch.zeros(logits.shape)
    for idx, logit in enumerate(logits):
        if (logit < threshold).all():
            continue

        output[idx] = (logit > threshold) 
    return output

