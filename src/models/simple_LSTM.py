import torch.nn as nn
import torch

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate = 0, bidirectional=False):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size = input_size, 
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            dropout = dropout_rate,
                            bidirectional = bidirectional,
                            batch_first=True)
        
        self.fc = nn.Sequential()

        # Create fully connected
        current_size = hidden_size * (self.bidirectional+1)
        while current_size > 8:
            self.fc.append(nn.Linear(current_size, current_size//2))
            self.fc.append(nn.LeakyReLU())
            current_size = current_size // 2

        self.fc.append(nn.Linear(current_size, 1))
                                    

    def forward(self, input):
        if len(input.shape) > 2:
            h0 = torch.randn(self.num_layers * (self.bidirectional+1), input.shape[0], self.hidden_size).to(input.device)
            c0 = torch.randn(self.num_layers * (self.bidirectional+1), input.shape[0], self.hidden_size).to(input.device)
        else :
            h0 = torch.randn(self.num_layers * (self.bidirectional+1), self.hidden_size).to(input.device)
            c0 = torch.randn(self.num_layers * (self.bidirectional+1), self.hidden_size).to(input.device)

        #print(f'input.shape={input.shape}')
        output, _ = self.lstm(input, (h0,c0)) # Input : (batch, seq size, input_size) , output : (batch, seq size, hidden size)
        # print(f'after lstm output.shape={output.shape}')
        output = self.fc(output)
        # print(f'after fc output.shape={output.shape}')
        output = output.squeeze(-1) # NOTE : No sigmoid, use LogitsLoss !
        # print(f'output.shape={output.shape}')
        return output
    
    def reset_weights(self):
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()