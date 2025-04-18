

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        last_outputs = output[range(len(output)), lengths - 1, :]
        out = self.fc(last_outputs)
        return out


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Sort the batch in the descending order
    sorted_batch = sorted(batch, key=lambda x: x[2], reverse=True)
    sequences, labels, lengths = zip(*sorted_batch)
    # Pad the sequences
    sequences_padded = pad_sequence(sequences, batch_first=True)
    return sequences_padded, torch.tensor(labels), torch.tensor(lengths)

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], len(self.sequences[idx])

# Initialize the model
model = LSTMClassifier(input_size=10, hidden_size=50, num_classes=2)

# Check if GPU is available and if so, move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Suppose we have some training data
sequences = [torch.randn(5, 10), torch.randn(3, 10), torch.randn(6, 10)]  # List of sequences
labels = torch.tensor([0, 1, 0])  # Class labels

# Create a DataLoader
dataset = SequenceDataset(sequences, labels)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Training loop
for epoch in range(100):  # Number of epochs
    for sequences, labels, lengths in data_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        #lengths = lengths.to(device)

        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')




