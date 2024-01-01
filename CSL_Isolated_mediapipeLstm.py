import os
import sys
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from models.ConvLSTM import CRNN, ResCRNN,ISL_mediapipe_lstm
from dataset import CSL_Isolated,CSL_Isolated_mediapipe
from train import train_epoch
from validation import val_epoch,test_epoch
from ConvLSTM import ISL_mediapipe_lstm

# Path setting
keypoints_path = "/root/autodl-tmp/ISL_keypoints"


log_path = "log/mediapipe_lstm_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
sum_path = "runs/islr_mediapipe_lstm_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())



#data_path = "datasets/CSL_Continuous/color"
label_path = "/root/autodl-tmp/ISLR/dictionary.txt"

model_path = "/root/autodl-tmp/models/ISLR_mediapipe_lstm"




# Log to file & tensorboard writer
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
logger.info('Logging to file...')
writer = SummaryWriter(sum_path)

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
epochs = 500
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-5
log_interval = 20
sample_size = 128
sample_duration = 16
num_classes = 100
lstm_hidden_size = 512
lstm_num_layers = 1
attention = False

# Train with Conv+LSTM
if __name__ == '__main__':
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = CSL_Isolated_mediapipe(keypoints_path=keypoints_path, label_path=label_path,
        num_classes=num_classes, mode="train")
    val_set = CSL_Isolated_mediapipe(keypoints_path=keypoints_path, label_path=label_path,
        num_classes=num_classes, mode="val")
    test_set = CSL_Isolated_mediapipe(keypoints_path=keypoints_path, label_path=label_path,
        num_classes=num_classes, mode="test")

    logger.info("Dataset samples: {}".format(len(train_set)+len(val_set)+len(test_set)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
    # Create model
    # model = CRNN(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes,
    #             lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers).to(device)

    # ============================
    model = ISL_mediapipe_lstm(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes,
                lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers, attention=attention).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Start training
    logger.info("Training Started".center(60, '#'))
    for epoch in range(epochs):
        # Train the model
        train_epoch(model, criterion, optimizer, train_loader, device, epoch, logger, log_interval, writer)

        # Validate the model
        val_epoch(model, criterion, val_loader, device, epoch, logger, writer)
        test_epoch(model, criterion, test_loader, device, epoch, logger, writer)
        # Save model
        torch.save(model.state_dict(), os.path.join(model_path, "slr_mediapipelstm_epoch{:03d}.pth".format(epoch+1)))
        logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    logger.info("Training Finished".center(60, '#'))
