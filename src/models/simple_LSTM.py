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
        
        self.dropout = nn.Dropout()
        
        self.fc = nn.Sequential()
        current_size = hidden_size
        while current_size > 8:
            self.fc.append(nn.Linear(current_size, current_size//2))
            self.fc.append(nn.LeakyReLU())
            current_size = current_size // 2
        self.fc.append(nn.Linear(current_size, 1))
        
        # self.fc = nn.Linear(hidden_size, 1)
                            

    def forward(self, input):
        # print(f'input.shape={input.shape}')
        output, _ = self.lstm(input) # Input : (batch, seq size, input_size) , output : (batch, seq size, hidden size)
        # print(f'after lstm output.shape={output.shape}')
        # output = self.dropout(output)
        output = self.fc(output)
        # print(f'after fc output.shape={output.shape}')
        output = output.squeeze(-1) # NOTE : No sigmoid, use LogitsLoss !
        # print(f'output.shape={output.shape}')
        return output