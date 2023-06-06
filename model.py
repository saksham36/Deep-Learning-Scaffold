'''
Written by: Saksham Consul 04/28/2023
Model for CNN + GRU + MLP + Softmax
'''

import torch.nn as nn
import torch
import pdb


class GRU_MLP_Softmax(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0):
        super(GRU_MLP_Softmax, self).__init__()
        # Defining the number of layers and the nodes in each layer
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # CNN layer
        self.cnn = nn.Conv2d(
            in_channels=3, out_channels=1, kernel_size=(1, 1))
        # GRU layers
        self.gru = nn.GRU(
            3*input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_prob
        )

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, output_dim),
            nn.GELU()
        )
        self.batch_norm = nn.BatchNorm2d(10)
        # self.softmax = nn.Softmax(dim=2) # CrossEntropyLoss incorporates this already

    def forward(self, x):
        """
        Takes an input tensor x of shape (batch_size, sequence_length (n_words), n_channels, input_dim)
        and passes it through the GRU layer to obtain a sequence of hidden states.
        We then take the last hidden state and pass it through the MLP layer to obtain a higher-level
        representation, and then pass that through the softmax layer to obtain the output probability vector.
        """
        batch_size, sequence_len, n_channel, input_dim = x.size()
        # Add batch norm
        x = self.batch_norm(x)
        x_view = x.view(batch_size, sequence_len, n_channel * input_dim)

        # (n_channel, batch_size*sequence_len, input_dim)
        # x_channel = self.cnn(x_view.transpose(0, 1))
        # x_channel = x_channel.view(batch_size, sequence_len, input_dim)
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_dim).to(x.device)
        import pdb
        pdb.set_trace()
        # Forward propagation by passing in the input and hidden state into the model
        # (batch_size, seq length, hidden_size)
        out, _ = self.gru(x_view, h0)
        # # Convert the final state to our desired output shape (batch_size, seq length, output_dim)
        out = self.mlp(out)  # (batch_size, seq length, output_dim)
        out = out.view(batch_size*sequence_len, self.output_dim)
        return out
