import torch.nn as nn
import torch

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate = 0):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, 
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            dropout = dropout_rate,
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, input):
        
        output, _ = self.lstm(torch.swapaxes(input, -1,-2)) # (batch, seq size, input_size)
        output = self.fc(output)
        output = output.squeeze()
        output = torch.sigmoid(output)

        return output