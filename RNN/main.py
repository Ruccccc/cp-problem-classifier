from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger

from models import RNN, LSTM, train, validate, predict
from utils import load_dataset, TrainDataset, collate_fn, plotloss

TRAIN_LENGTH = 0.8
BATCH_SIZE = 32
EPOCH = 50
LEARNING_RATE = 0.005

"""
Load Dataset
"""

logger.info("Load Dataset")

desc, labels, voc_dict, label_dict = load_dataset('dataset/problems1.csv')
desc, labels = shuffle(desc, labels, random_state=777)

train_len = int(len(desc) * TRAIN_LENGTH)
train_dataset = TrainDataset(desc[:train_len], labels[:train_len])
val_dataset = TrainDataset(desc[train_len:], labels[train_len:])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

"""
RNN - train
"""

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

logger.info(f"RNN - Train with {device}")

# model
model = RNN(len(voc_dict), 100, 256, len(label_dict), 1).to(device)
# model = LSTM(len(voc_dict), 100, 256, len(label_dict), 1).to(device)

# loss function and optimizer
pos_weight = [0] * len(label_dict)
for i in labels:
    for j in range(len(label_dict)):
        pos_weight[j] += i[j]
pos_weight = torch.tensor(pos_weight)
pos_weight = (len(desc) - pos_weight) / pos_weight
pos_weight = pos_weight.clamp(max=20.0).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
val_losses = []
max_F1 = 0
best_thres = 0
try:
    for epoch in range(EPOCH):

        train_loss = train(model, train_loader, criterion, optimizer, device)

        curr_max_f1 = 0
        curr_best_thres = 0
        for t in [i/100 for i in range(20, 80, 5)]:
            val_loss, acc, f1_micro = validate(model, val_loader, criterion, device, t)
            if f1_micro > curr_max_f1:
                curr_max_f1 = f1_micro
                curr_best_thres = t

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if max_F1 < curr_max_f1:
            max_F1 = curr_max_f1
            best_thres = curr_best_thres

        logger.info(f"Epoch {epoch:>2} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f} | F1 score: {curr_max_f1:.6f} | Best Thres: {curr_best_thres}")
except KeyboardInterrupt:
    pass

logger.info(f"Maximum F1 micro: {max_F1}")
logger.info(f"Best Threshold: {best_thres}")
