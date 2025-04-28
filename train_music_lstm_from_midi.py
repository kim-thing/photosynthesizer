# train_music_lstm_from_midi.py
import torch
import torch.nn as nn
import os
import random
from mido import MidiFile
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from src.lstm_model import MusicLSTM

# === SETTINGS ===
MIDI_ROOT = "large_music"  # 👈✅ YOUR MIDI folder here!
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
NUM_EPOCHS = 10
SAMPLE_SIZE = 100  # Pick 100 random MIDI files

# === Step 1: Extract notes from MIDI ===
def extract_notes_from_midi(midi_path):
    notes = []
    try:
        midi = MidiFile(midi_path)
        for track in midi.tracks:
            for msg in track:
                if not msg.is_meta and msg.type == 'note_on' and msg.velocity > 0:
                    notes.append(msg.note)
    except Exception as e:
        print(f"⚠️ Error reading {midi_path}: {e}")
    return notes

# === Step 2: Randomly pick MIDI files ===
all_midi_files = []
for root, _, files in os.walk(MIDI_ROOT):
    for file in files:
        if file.endswith(".mid") or file.endswith(".midi"):
            all_midi_files.append(os.path.join(root, file))

if len(all_midi_files) < SAMPLE_SIZE:
    raise ValueError(f"Not enough MIDI files found! Only {len(all_midi_files)} available.")

sampled_files = random.sample(all_midi_files, SAMPLE_SIZE)

# === Step 3: Build note dataset ===
all_notes = []
for midi_path in sampled_files:
    notes = extract_notes_from_midi(midi_path)
    if len(notes) >= SEQUENCE_LENGTH + 1:
        all_notes.extend(notes)

print(f"✅ Collected {len(all_notes)} total notes.")

# === Step 4: Prepare (X, y) sequences ===
X = []
y = []

for i in range(len(all_notes) - SEQUENCE_LENGTH):
    seq_in = all_notes[i:i+SEQUENCE_LENGTH]
    seq_out = all_notes[i+SEQUENCE_LENGTH]
    X.append(seq_in)
    y.append(seq_out)

X = np.array(X) / 127.0  # Normalize MIDI note values (0-1)
y = np.array(y) / 127.0

X = torch.tensor(X).float().unsqueeze(-1)  # (samples, seq_len, 1)
y = torch.tensor(y).float().unsqueeze(-1)  # (samples, 1)

print(f"✅ Final training data shape: {X.shape}")

# === Step 5: Train MusicLSTM ===
model = MusicLSTM(input_size=1, hidden_size=128, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    preds = []
    trues = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds.append((output > 0.5).int())
        trues.append((yb > 0.5).int())

    preds = torch.cat(preds)
    trues = torch.cat(trues)
    acc = accuracy_score(trues.cpu().numpy().flatten(), preds.cpu().numpy().flatten())
    f1 = f1_score(trues.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), zero_division=0)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {total_loss/len(train_loader):.4f} - Acc: {acc:.4f} - F1: {f1:.4f}")

# === Step 6: Save model ===
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/music_lstm.pt")
print("✅ Trained Music LSTM saved to models/music_lstm.pt")
