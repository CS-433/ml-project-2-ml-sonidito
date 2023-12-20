[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13065283&assignment_repo_type=AssignmentRepo)

# ML4Science : Image processing/pattern recognition on MHD spectrograms to automate the detection of phase in the discharge characterized by Magneto-HydroDynamic instabilities

## Getting Start

Requirements :

- Ananconda
- PyTorch with CUDA adapted to your machine

The conda environement can be create by runing this command

```text
conda env create -f environment.yml
```

## Training

### LSTM

```text
python .\train.py .\data\ 32 1 0.001 --weight_decay=1e-3 --l1_sigma=1e-4 --dropout_rate=0 --patience=10 --delta=1e-3 --model_path=models --n_epoch=1000 --batch_size=128 --max_length=4293 
```

## Testing

### LSTM

```text
python .\test.py .\models\lstm.pt .\data\ --max_length=4293 --batch_size=128
```

## Repo structure

The repo doesn't contain the dataset; it is the user's responsability to palce the data according to the specified structure.

```text
├───data
│   ├───dataset_h5
│   ├───dataset_pickle
│   └───MHD_labels
├───models          # Contain the weight of the train model
├───notebook
└───src
    ├───data
    └───models
```

