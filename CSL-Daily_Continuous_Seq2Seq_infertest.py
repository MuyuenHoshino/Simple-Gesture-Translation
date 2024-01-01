import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from dataset_new import CSL_Daily_Continuous_TwoStream,CSL_Daily_Continuous_TwoStream_gloss_simple,CSL_Daily_Continuous_TwoStream_gloss,CSL_Daily_Continuous_TwoStream_balanced,CSL_Daily_Continuous_TwoStream_256,CSL_Daily_Continuous_TwoStream_256_balanced,CSL_Daily_Continuous_TwoStream_real,CSL_Daily_Continuous_TwoStream_real_balanced
from models.Seq2Seq import Encoder, Decoder, Seq2Seq,Myencoder_mediapipe,Mediapipe_Transformer,TwoStream_Transformer, CSL_Daily_TwoStream_Transformer
from train import train_seq2seq,train_seq2seq_fuse,train_seq2seq_fuse_old,train_seq2seq_fuse_greedy
from validation import val_seq2seq,test_seq2seq,val_seq2seq_fuse,test_seq2seq_fuse,val_seq2seq_fuse_daily,test_seq2seq_fuse_daily

# Path setting
data_path = "/root/autodl-tmp/picture_new"
pkl_path = "/root/autodl-tmp/sentence_label/csl2020ct_v2.pkl"
keypoints_path = "/root/autodl-tmp/keypoints_new"
corpus_path = "/root/autodl-tmp/sentence_label/video_map.txt"
# model_path = "/root/autodl-tmp/models/seq2seq_models_mediapipe_500epoch_720-720_new"

model_path = "/root/autodl-tmp/models/CSL-Daily_TwoStream"
log_path = "log/CSL-Daily_TwoStream_{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now())
sum_path = "runs/CSL-Daily_TwoStream_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())

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
batch_size = 16
learning_rate = 1e-5


weight_decay = 1e-5
sample_size = 256
sample_duration = 64
enc_hid_dim = 512
emb_dim = 256
dec_hid_dim = 512
dropout = 0.5
clip = 1
log_interval = 40

if __name__ == '__main__':
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    # train_set = CSL_Continuous(data_path=data_path, dict_path=dict_path,
    #     corpus_path=corpus_path, frames=sample_duration, train=True, transform=transform)
    # val_set = CSL_Continuous(data_path=data_path, dict_path=dict_path,
    #     corpus_path=corpus_path, frames=sample_duration, train=False, transform=transform)


    # test_set = CSL_Continuous_Mediapipe(data_path=data_path, dict_path=dict_path,
    #    corpus_path=corpus_path, frames=sample_duration, mode="test")
    # train_set = CSL_Continuous_Mediapipe(data_path=data_path, dict_path=dict_path,
    #     corpus_path=corpus_path, frames=sample_duration, mode="train")

    # val_set = CSL_Continuous_Mediapipe(data_path=data_path, dict_path=dict_path,
    #     corpus_path=corpus_path, frames=sample_duration, mode="val")
    # test_set = CSL_Continuous_Mediapipe(data_path=data_path, dict_path=dict_path,
    #     corpus_path=corpus_path, frames=sample_duration, mode="test")

    
    #train_set = CSL_Daily_Continuous_TwoStream_real(data_path=data_path, pkl_path=pkl_path,
    #    corpus_path=corpus_path, frames=sample_duration, mode="train",keypoints_path = keypoints_path,transform = transform)
    val_set = CSL_Daily_Continuous_TwoStream_real(data_path=data_path, pkl_path=pkl_path,
        corpus_path=corpus_path, frames=sample_duration, mode="val",keypoints_path = keypoints_path,transform = transform)
    test_set = CSL_Daily_Continuous_TwoStream_real(data_path=data_path, pkl_path=pkl_path,
        corpus_path=corpus_path, frames=sample_duration, mode="test",keypoints_path = keypoints_path,transform = transform)
    

    total_dict = test_set.dict

    # train_set = CSL_Continuous_Char(data_path=data_path, corpus_path=corpus_path,
    #     frames=sample_duration, train=True, transform=transform)
    # val_set = CSL_Continuous_Char(data_path=data_path, corpus_path=corpus_path,
    #     frames=sample_duration, train=False, transform=transform)

    logger.info("Dataset samples: {}".format(len(val_set)+len(test_set)))

    # logger.info("Dataset samples: {}".format(len(train_set)+len(val_set)))
    # logger.info("Dataset samples: {}".format(len(train_set)))
    #train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,drop_last=False)
    # Create Model
    #encoder = Myencoder_mediapipe().to(device)

    # encoder = Encoder().to(device)
    # decoder = Decoder(output_dim=train_set.output_dim, emb_dim=emb_dim, enc_hid_dim=enc_hid_dim, dec_hid_dim=dec_hid_dim, dropout=dropout).to(device)
    # model = Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)
    
    #model = Mediapipe_Transformer().to(device)

    model = CSL_Daily_TwoStream_Transformer().to(device)
    load_path = "/root/autodl-tmp/models/CSL-Daily_TwoStream/CSL-Daily_TwoStream_epoch031.pth"
    model.load_state_dict(torch.load(load_path)) 
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, foreach=None)
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2, momentum=0.99)
    #optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, lr_decay=0, weight_decay=0)
    #optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    # Start training
    logger.info("Training Started".center(60, '#'))
    for epoch in range(epochs):
        # # Train the model
        # train_seq2seq(model, criterion, optimizer, clip, train_loader, device, epoch, logger, log_interval, writer)

        # # Validate the model
        # val_seq2seq(model, criterion, val_loader, device, epoch, logger, writer)

        # test_seq2seq(model, criterion, test_loader, device, epoch, logger, writer)
        #test_seq2seq_fuse_daily(model, criterion, test_loader, device, epoch, logger, writer, total_dict)

        #train_seq2seq_fuse_old(model, criterion, optimizer, clip, train_loader, device, epoch, logger, log_interval, writer, total_dict)
        #train_seq2seq_fuse_greedy(model, criterion, optimizer, clip, train_loader, device, epoch, logger, log_interval, writer,total_dict)
        val_seq2seq_fuse_daily(model, criterion, val_loader, device, epoch, logger, writer, total_dict)
        test_seq2seq_fuse_daily(model, criterion, test_loader, device, epoch, logger, writer, total_dict)

        # Save model
        torch.save(model.state_dict(), os.path.join(model_path, "CSL-Daily_TwoStream_epoch{:03d}.pth".format(epoch+1)))
        logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

    logger.info("Training Finished".center(60, '#'))
