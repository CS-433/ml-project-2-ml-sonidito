import torch.nn as nn
import torch

class SimpleLSTM(nn.Module):
    """
        Curstom LSTM followed by a fully connected net
    
        Attributes
            input_size : int
                The input size of the LSTM
            hidden_size : int
                The hidden size of the LSTM
            num_layers : int
                Thu number of layer of the LSTM
            bidirectional : bool
                Set the directionality of the LSTM
            lstm : torch.nn
                LSTM module
            fc : nn.Sequential
                Fully connected net, create Linear module sequentially, the output size of the Lienar is always the
                half of the input size. Start with the hidden size and stop when the input size is smaller than 8.
                The last layer a output size of 1. Each Linear layer is followed by a leaky ReLU
    """

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
        
        self.bn = nn.BatchNorm1d(hidden_size * (self.bidirectional+1))
        
        self.fc = nn.Sequential()

        # Create fully connected
        current_size = hidden_size * (self.bidirectional+1)
        while current_size > 8:
            self.fc.append(nn.Linear(current_size, current_size//2))
            self.fc.append(nn.LeakyReLU())
            current_size = current_size // 2

        self.fc.append(nn.Linear(current_size, 1))
                                    

    def forward(self, input):
        # Random init h0 and c0 input
        if len(input.shape) > 2:
            h0 = torch.randn(self.num_layers * (self.bidirectional+1), input.shape[0], self.hidden_size).to(input.device)
            c0 = torch.randn(self.num_layers * (self.bidirectional+1), input.shape[0], self.hidden_size).to(input.device)
        else :
            h0 = torch.randn(self.num_layers * (self.bidirectional+1), self.hidden_size).to(input.device)
            c0 = torch.randn(self.num_layers * (self.bidirectional+1), self.hidden_size).to(input.device)

        output, _ = self.lstm(input, (h0,c0)) # output = (batch, seq, hidden size)
        output = self.bn(output.permute(0,2,1)).permute(0,2,1)
        output = self.fc(output)
        output = output.squeeze(-1) # NOTE : No sigmoid, use LogitsLoss as criterion!
        return output
    
    def reset_weights(self):
        """
            Reset all layers that can be reset
        """

        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()