import numpy as np
import matplotlib.pyplot as plt
import os
import hickle
import torch
from tqdm.notebook import tqdm
from metrics import compute_metrics


def evaluate_model(model, test_loader, device, threshold=0.50):
    """
    Evaluates the model on the test set.

    Parameters:
    - model: The trained model to evaluate.
    - test_loader: The DataLoader object containing the test data.
    - device: The device where the model and data are located.
    - threshold: The threshold for classifying a slice as positive.

    Returns:
    - accuracy: The accuracy of the model.
    - f1: The F1 score of the model.
    - kappa: Cohen's kappa for the model.
    """
    model.eval()
    with torch.no_grad():
        logits = []
        labels = []
        with tqdm(test_loader, unit='batch', desc='Evaluating') as tepoch:
            for batch in tepoch:
                x_batch = batch['window_odd'].to(device)
                y_batch = batch['label'].to(device)
                output = model(x_batch)
                logits.append(output)
                labels.append(y_batch)

        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)

        accuracy, f1, kappa, _ = compute_metrics(logits, labels, threshold)

    return accuracy, f1, kappa


def plot_spectrogram_slices(test_loader, shotnos=[], cmap='jet', figsize=(16, 16)):
    """
    Plots the pre-processed spectrogram slices from the test set, optionally filtered by provided shot numbers.
    Plots for all shot numbers by default.

    Parameters:
    - test_loader: The test set data loader.
    - shotnos: List of shot numbers to plot.
    - cmap: The colormap to use.
    - figsize: The size of the figure.
    """
    # Create a dictionary to hold data grouped by shot number
    shotno_dict = {}
    for i in range(len(test_loader.dataset)):
        try:
            data = test_loader.dataset[i]
            shotno = data['shotno']
            if shotno not in shotno_dict:
                shotno_dict[shotno] = []
            shotno_dict[shotno].append(data)
        except KeyError as e:
            print(f"An error occurred: {e}")
            continue

    # Determine which shot numbers to plot
    shotnos_to_plot = shotnos if shotnos else shotno_dict.keys()

    # Iterate over the selected shot numbers
    for shotno in shotnos_to_plot:
        shot_data = shotno_dict.get(shotno, [])
        if not shot_data:
            print(f"No data found for shot number {shotno}.")
            continue

        num_slices = len(shot_data)
        num_subplots_side = int(np.ceil(np.sqrt(num_slices)))

        fig, axes = plt.subplots(num_subplots_side, num_subplots_side, figsize=figsize)
        fig.suptitle(f'Pre-processed Odd-N Spectrogram Slices \nShot #{shotno}', fontsize=16)

        for i, ax in enumerate(axes.flat):
            if i < num_slices:
                spectrogram_slice = np.array(shot_data[i]['window_odd'][1])
                timestamps = shot_data[i]['time']
                freqs = shot_data[i]['frequency'] / 1000  # Convert to kHz
                img = np.clip(spectrogram_slice, 0, 255)
                aspect_ratio = 'auto'
                im = ax.imshow(img, cmap=cmap, aspect=aspect_ratio, origin='lower',
                               extent=[timestamps[0], timestamps[-1], freqs[0], freqs[-1]])
                ax.set_title(f'Slice {i}', fontsize=8)
                ax.set_xlabel('Time [s]', fontsize=6)
                ax.set_ylabel('Frequency [kHz]', fontsize=6)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()


def plot_spectrograms(shotnos, data_path, file_ext, db_thresh=-100):
    """
    Plots the spectrograms for a list of shot numbers.

    Parameters:
    - shotnos: List of shot numbers to plot.
    - data_path: Path to the directory containing the data files.
    - file_ext: File extension of the data files.
    """

    for shotno in shotnos:
        # Load the shot data
        data_shot = load_shot(shotno, data_path, file_ext)
        if data_shot is None:
            continue

        # Extracting inputs
        inputs = data_shot["x"]["spectrogram"]
        spec_odd = np.clip(inputs["OddN"], a_min=-90, a_max=0)
        f = inputs["frequency"]
        t = inputs["time"]

        # Plotting the spectrogram
        fig, ax = plt.subplots()
        ax.imshow(spec_odd.T, extent=(t[0], t[-1], f[0], f[-1]), aspect='auto', cmap='jet', origin='lower')
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(f[0], f[-1])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        fig.set_dpi(125)
        plt.title(f"Odd-N Spectrogram \nShot #{shotno}")

        plt.show()


def load_shot(shotno, data_path, file_ext):
    """
    Loads the shot data from a file.
    """
    file_path = os.path.join(data_path, f"{shotno}.{file_ext}")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    with open(file_path, "rb") as f:
        return hickle.load(f)


def test_model_for_shot(model, test_loader, device, shotno, threshold=0.50):
    """
    Test the model for a specific shot number

    Parameters:
    model (torch.nn.Module): The trained model to test.
    test_loader (torch.utils.data.DataLoader): The DataLoader object containing the test data.
    device (torch.device): The device where the model and data are located.
    shotno (int): The shot number to test.
    threshold (float): The threshold for classifying a slice as positive.

    Returns:
    predictions (numpy.ndarray): The model's predictions for the shot number.
    labels (numpy.ndarray): The true labels for the shot number.
    """
    model.eval()
    for batch in test_loader:
        # Convert batch['shotno'] to a list or numpy array for comparison
        shotnos_batch = batch['shotno'].cpu().numpy()

        # Check if the current batch contains the specified shot number
        if shotno in shotnos_batch:
            # Find the index/indices of the specified shot number in the batch
            indices = [i for i, s in enumerate(shotnos_batch) if s == shotno]
            with torch.no_grad():
                x_batch = batch['window_odd'][indices].to(device)
                output = model(x_batch)

                print(f"Processing {output.shape[0]} slices for shot number {shotno}")
                predictions = (torch.sigmoid(output) > threshold).squeeze().cpu().numpy()
                labels = batch['label'][indices].squeeze().cpu().numpy()

                return predictions.astype(int), labels.astype(int)
    raise ValueError(f"No data found for shot #{shotno}.")


def plot_spectrogram_with_predictions(shotnos, data_path, file_ext, model, test_loader, device, window_size, overlap,
                                      threshold=0.50, time_step=5.12e-4, cmap='jet'):
    for shotno in shotnos:
        # Load the shot data
        data_shot = load_shot(shotno, data_path, file_ext)
        if data_shot is None:
            print(f"No data found for shot number {shotno}.")
            continue

        # Extracting inputs
        inputs = data_shot["x"]["spectrogram"]
        spec = np.clip(inputs["OddN"], a_min=-90, a_max=0)
        frequency = inputs["frequency"]
        time = inputs["time"]

        # Get predictions and labels
        predictions, labels = test_model_for_shot(model, test_loader, device, shotno, threshold)

        fig, ax = plt.subplots(figsize=(9, 5))
        cax = ax.imshow(spec.T, extent=(time[0], time[-1], frequency[0], frequency[-1]), aspect='auto', cmap='jet',
                        origin='lower')
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(frequency[0], frequency[-1])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        fig.set_dpi(150)
        plt.title(f"Odd-N Spectrogram \nShot #{shotno}")

        # Initialize legends
        legend_handles = []

        # Apply a translucent mask over slices where the prediction is True (merging consecutive slices)
        start_time = None
        actual_window_size = window_size * (1 - overlap)  # Adjust window size for overlap

        for i, prediction in enumerate(predictions):
            slice_start_time = i * actual_window_size * time_step
            slice_end_time = slice_start_time + window_size * time_step

            if prediction:
                if start_time is None:
                    # Start of a new true prediction span
                    start_time = slice_start_time
                # If it's the last prediction or there's no overlap with the next, finalize the span
                if i == len(predictions) - 1 or not predictions[i + 1]:
                    end_time = slice_end_time
                    pred_patch = ax.axvspan(start_time, end_time, color='red', alpha=0.3)
                    start_time = None
                    if len(legend_handles) == 0:  # Add the legend only once
                        legend_handles.append(pred_patch)
            else:
                start_time = None  # Reset the start time for the next span

        # Process ground truth
        start_time = None
        for i, label in enumerate(labels):
            slice_start_time = i * actual_window_size * time_step
            slice_end_time = slice_start_time + window_size * time_step

            if label:
                if start_time is None:
                    start_time = slice_start_time  # Start of a new true span
                if i == len(labels) - 1 or not labels[i + 1]:
                    end_time = slice_end_time  # End of a true span
                    gt_line_start = ax.axvline(x=start_time, color='red', linestyle='--', linewidth=2)
                    gt_line_end = ax.axvline(x=end_time, color='red', linestyle='--', linewidth=2)
                    start_time = None
                    if 'Ground Truth' not in [lh.get_label() for lh in legend_handles]:
                        legend_handles.append(gt_line_start)
            else:
                start_time = None

        # Add legend with unique handles
        ax.legend(handles=legend_handles, labels=['Prediction', 'Ground Truth'], loc='best')

        fig.colorbar(cax, ax=ax, orientation='vertical', label='Attenuation [dB]')
        plt.show()
