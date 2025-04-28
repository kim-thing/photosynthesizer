import torch
import torch.nn as nn

class MusicLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1):
        super(MusicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass input through LSTM
        out, _ = self.lstm(x)  # out shape: (batch_size, seq_len, hidden_size)

        # Only take the output from the LAST time step
        out = out[:, -1, :]  # shape becomes (batch_size, hidden_size)

        # Pass through the final fully connected layer
        out = self.fc(out)   # shape becomes (batch_size, output_size)

        return out
