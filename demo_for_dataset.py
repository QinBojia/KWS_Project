# python
import os
import glob
import random
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

root = r'C:\Users\m1339\PycharmProjects\KWS_Project\dataset\SpeechCommands\speech_commands_v0.02'
target_words = ["four", "five", "off", "forward"]
batch_size, epochs, lr = 64, 5, 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpeechCommandsLocal(Dataset):
    def __init__(self, root_dir: str, subset: str, sample_rate: int = 16000, max_len: int = 16000):
        self.sr = sample_rate
        self.max_len = max_len

        val_list_all = self._read_list(root_dir, "validation_list.txt")
        test_list_all = self._read_list(root_dir, "testing_list.txt")

        # keep only target classes
        def is_target(p):
            return os.path.basename(os.path.dirname(p)) in target_words

        val_list = [p for p in val_list_all if is_target(p)]
        test_list = [p for p in test_list_all if is_target(p)]

        all_files = [p for w in target_words for p in glob.glob(os.path.join(root_dir, w, "*.wav"))]

        if subset == "validation":
            files = val_list
        elif subset == "testing":
            files = test_list
        else:  # training
            exclude = set(os.path.relpath(p, root_dir).replace("\\", "/") for p in (val_list + test_list))
            files = [p for p in all_files if os.path.relpath(p, root_dir).replace("\\", "/") not in exclude]

        self.files = files

    def _read_list(self, root_dir, filename) -> List[str]:
        path = os.path.join(root_dir, filename)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return [os.path.join(root_dir, line.strip()) for line in f]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = os.path.basename(os.path.dirname(path))
        # safety: skip any non-target sample
        if label not in target_words:
            # pick another index deterministically
            return self.__getitem__((idx + 1) % len(self.files))

        audio, _ = librosa.load(path, sr=self.sr)
        audio = self._pad_or_trim(audio)
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=40, n_fft=400, hop_length=160, n_mels=64
        )
        mfcc = torch.tensor(mfcc, dtype=torch.float32)
        return mfcc.unsqueeze(0), target_words.index(label)

    def _pad_or_trim(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) > self.max_len:
            return audio[: self.max_len]
        if len(audio) < self.max_len:
            return np.pad(audio, (0, self.max_len - len(audio)))
        return audio

class CNNKWS(nn.Module):
    def __init__(self, num_classes=len(target_words)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, 20, T//2)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, 10, T//4)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def make_loader(split: str) -> DataLoader:
    ds = SpeechCommandsLocal(root, subset=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "training"))

def run_epoch(loader, model, opt, criterion, train: bool):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            opt.step()
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)

train_loader = make_loader("training")
val_loader = make_loader("validation")
test_loader = make_loader("testing")

model = CNNKWS().to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, epochs + 1):
    train_loss, train_acc = run_epoch(train_loader, model, opt, criterion, True)
    val_loss, val_acc = run_epoch(val_loader, model, opt, criterion, False)
    print(f"Epoch {epoch}: train_loss={train_loss:.3f} acc={train_acc:.3f} | val_loss={val_loss:.3f} acc={val_acc:.3f}")

test_loss, test_acc = run_epoch(test_loader, model, opt, criterion, False)
print(f"Test: loss={test_loss:.3f} acc={test_acc:.3f}")
