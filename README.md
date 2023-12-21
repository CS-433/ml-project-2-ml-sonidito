[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13065283&assignment_repo_type=AssignmentRepo)

# ML4Science : Image processing/pattern recognition on MHD spectrograms to automate the detection of phase in the discharge characterized by Magneto-HydroDynamic instabilities

## Getting Started

Requirements :

- Anaconda
- PyTorch with CUDA adapted to your machine

The conda environment can be created by running this command

```text
conda env create -f environment.yml
```


## LSTM

For the LSTM model we provide two scripts, one to train the model and another one to do inference

### Training

Running this command will generate the same model saved the `models\` folder

```text
python .\train_LSTM.py .\data\ 128 1 0.01 --weight_decay=1e-5 --l1_sigma=1e-4 --dropout_rate=0 --patience=0 --delta=1e-3 --model_path=models --n_epoch=400 --batch_size=128 --max_length=4293 
```
### Inference

To use our model to predict with new data use this command. Make sure to adapt the path `.\data\`

```text
python .\test_LSTM.py .\models\lstm.pt .\data\ --max_length=4293 --batch_size=128
```

## CNNs

The CNN models training and evaluation pipelines are contained in their respective notebooks in `notebooks/resnet18_CNN.ipynb` and `notebooks/efficientnet_CNN.ipynb`. 
The cross-validation process for the EfficientNet-B0 model is done in `notebooks/cross_validation.ipynb`

## Repo structure

The repo doesn't contain the dataset; it is the user's responsibility to place the data according to the specified structure.
The dataset is available through this [SwitchDrive link](https://drive.switch.ch/index.php/s/K7BYcTRIZMupM7T).

```text
├───data
│   ├───dataset_h5
│   ├───dataset_pickle
│   └───MHD_labels
├───models          # Contains the weights of the trained model
├───notebooks
└───src
    ├───data
    └───models
```

## Models

This repo contains 3 models, 2 CNNs and 1 LSTM. You can find below more information about our models

### Number of parameters

| **Model**           | **Number of Parameters** |
|---------------------|--------------------------|
| **LSTM**            | 78k                      |
| **EfficientNet-B0** | 4.8M                     |
| **ResNet-18**       | 11.3M                    |

### Hyperparameters

#### EfficientNet and ResNet

| **Learning Rate**    | **Dropout rate** | **Weight decay** | **Batch size** |
|----------------------|------------------|------------------|----------------|
| $7  \times 10^{-5}$ | 0.3              | 1                | 64             |

#### LSTM 

| **lr** | **Hidden size** | **Num layer** | **Weight decay**   | **L1 sigma**       | **Batch size** |
|--------|-----------------|---------------|--------------------|--------------------|----------------|
| $0.01$ | $128$           | $1$           | $1 \times 10^{-5}$ | $1 \times 10^{-4}$ | 128            |


## Results 

| **Model**                               | **Cohen's kappa** | **F1 score** |
|-----------------------------------------|-------------------|--------------|
| **LSTM**                                | $0.882$           | $0.900$      |
| **EfficientNet-B0 CNN**                 | $0.765$           | $0.828$      |
| **ResNet18 CNN**                        | $0.663$           | $0.758$      |



