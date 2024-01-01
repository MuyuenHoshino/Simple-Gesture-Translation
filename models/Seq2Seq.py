import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import random
import math
import os,inspect,sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)
from ConvLSTM import ResCRNN

"""
Implementation of Sequence to Sequence Model
Encoder: encode video spatial and temporal dynamics e.g. CNN+LSTM
Decoder: decode the compressed info from encoder
"""
class Encoder(nn.Module):
    def __init__(self, lstm_hidden_size=512, arch="resnet18"):
        super(Encoder, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size

        # network architecture
        if arch == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif arch == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif arch == "resnet50":
            resnet = models.resnet50(pretrained=True)
        elif arch == "resnet101":
            resnet = models.resnet101(pretrained=True)
        elif arch == "resnet152":
            resnet = models.resnet152(pretrained=True)
        # delete the last fc layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.lstm = nn.LSTM(
            input_size=resnet.fc.in_features,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
        )

    def forward(self, x):
        # CNN
        cnn_embed_seq = []
        # x: (batch_size, channel, t, h, w)
        for t in range(x.size(2)):
            # with torch.no_grad():
            out = self.resnet(x[:, :, t, :, :])
            # print(out.shape)
            out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)
        print(cnn_embed_seq.shape)
        # LSTM
        # use faster code paths
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(cnn_embed_seq, None)

        # num_layers * num_directions = 1
        return out, (h_n.squeeze(0), c_n.squeeze(0))


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim+enc_hid_dim, dec_hid_dim)
        self.fc = nn.Linear(emb_dim+enc_hid_dim+dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, context):

        input = input.unsqueeze(0)

        # embedded(1, batch_size, emb_dim): embed last prediction word
        embedded = self.dropout(self.embedding(input))

        # rnn_input(1, batch_size, emb_dim+enc_hide_dim): concat embedded and context 
        rnn_input = torch.cat((embedded, context.unsqueeze(0)), dim=2)

 
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))

        hidden = hidden.squeeze(0)
        cell = cell.squeeze(0)
        embedded = embedded.squeeze(0)

        # prediction
        prediction = self.fc(torch.cat((embedded, context, hidden), dim=1))

        return prediction, (hidden, cell)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, imgs, target, teacher_forcing_ratio=0.5):
        # imgs: (batch_size, channels, T, H, W)
        # target: (batch_size, trg len)
        batch_size = imgs.shape[0]
        trg_len = target.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs(batch, seq_len, hidden_size): all hidden states of input sequence
        encoder_outputs, (hidden, cell) = self.encoder(imgs)

        # compute context vector
        context = encoder_outputs.mean(dim=1)

        # first input to the decoder is the <sos> tokens
        input = target[:,0]

        for t in range(1, trg_len):
            # decode
            output, (hidden, cell) = self.decoder(input, hidden, cell, context)

            # store prediction
            outputs[t] = output

            # decide whether to do teacher foring
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token
            top1 = output.argmax(1)

            # apply teacher forcing
            input = target[:,t] if teacher_force else top1
        print("target",target.shape)
        #print(outputs.shape)
        return outputs

        
class Myencoder_mediapipe(nn.Module):
    def __init__(self):
        super(Myencoder_mediapipe, self).__init__()
        self.lstm = nn.LSTM(
            input_size=258,
            hidden_size=512,
            batch_first=True,
        )
    def forward(self, x):
        self.lstm.flatten_parameters()
        out, (h_n, c_n) = self.lstm(x, None)

        # num_layers * num_directions = 1
        return out, (h_n.squeeze(0), c_n.squeeze(0))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Mediapipe_Transformer(nn.Module):
        

    def __init__(self):
        super(Mediapipe_Transformer, self).__init__()
        d_model = 258  # Embedding Size
        d_ff = 2048  # FeedForward dimension
        n_layers = 6  # number of Encoder of Decoder Layer
        n_heads = 6  # number of heads in Multi-Head Attention
        self.transformer = nn.Transformer(d_model = d_model, nhead= n_heads,
                                          num_encoder_layers=n_layers, num_decoder_layers=n_layers,
                                          dim_feedforward=d_ff)
        self.positional_encoding = PositionalEncoding(258, dropout=0)
        self.tgt_embedding = nn.Embedding(507, 258, padding_idx=2)
        self.fc1 = nn.Linear(258, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 507)
        self.fc = nn.Linear(258, 507)
    def get_key_padding_mask(tokens):
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask

    def forward(self, x, target,teacher_forcing_ratio=0.5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #src_key_padding_mask = Mediapipe_Transformer.get_key_padding_mask(x)
        #tgt_key_padding_mask = Mediapipe_Transformer.get_key_padding_mask(target)
        tgt_key_padding_mask = target.data.eq(2).cuda()

        target = self.tgt_embedding(target)
        tgt_mask = self.transformer.generate_square_subsequent_mask(9).to(device)
        #print(x.shape)
        #x = torch.reshape(x,((48,-1,258)))
        
        #print(target.shape)
        #target = torch.reshape(target,((9,-1,258)))
        x = x.transpose(0,1)
        target = target.transpose(0,1)
        #print(target)
        x = self.positional_encoding(x)
        target = self.positional_encoding(target)

        #print(x.shape)
        print(target.shape)
        out = self.transformer(src=x,tgt=target,
                               tgt_mask=tgt_mask
                               )
        print(out.shape)
        # out = self.fc1(out)
        # out = self.fc2(out)
        # out = self.fc3(out)
        out = self.fc(out)
        #print(out)
        #out = out.reshape(9,16,507)
        #out=torch.reshape(out,((9,-1,507)))
        #out = out.transpose(0,1)
        #print(out)
        return out

class TwoStream_Transformer(nn.Module):
        

    def __init__(self):
        super(TwoStream_Transformer, self).__init__()
        d_model = 512#+258  # Embedding Size
        d_ff = 2048  # FeedForward dimension
        n_layers = 6  # number of Encoder of Decoder Layer
        n_heads = 8  # number of heads in Multi-Head Attention
        self.transformer = nn.Transformer(d_model = d_model, nhead= n_heads,
                                          num_encoder_layers=n_layers, num_decoder_layers=n_layers,
                                          dim_feedforward=d_ff)
        self.positional_encoding = PositionalEncoding(770, dropout=0)
        self.tgt_embedding = nn.Embedding(7753, 770, padding_idx=0)

        self.fc = nn.Linear(770, 7753)#7750+3
        
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.freeze(self.resnet)

    def freeze(layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False
        
    def get_key_padding_mask(tokens):
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask

    def forward(self,images, sequence,target,teacher_forcing_ratio=0.5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tgt_mask = self.transformer.generate_square_subsequent_mask(target.size(-1)).to(device)



        images = images.to(device)
        sequence = sequence.to(device)
        cnn_embed_seq = []
        # x: (batch_size, channel, t, h, w)
        for t in range(images.size(2)):
            # with torch.no_grad():
            out = self.resnet(images[:, :, t, :, :])
            # print(out.shape)
            out = out.view(out.size(0), -1)
            cnn_embed_seq.append(out)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).to(device)
        # batch first
        cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

        #print("cnn",cnn_embed_seq.shape)

        fuse = torch.cat((sequence, cnn_embed_seq), dim=2).to(device)
        #fuse = cnn_embed_seq
        #print("fuse",fuse.shape)


        tgt_key_padding_mask = target.data.eq(0).cuda()

        target = self.tgt_embedding(target)
        

        fuse= fuse.transpose(0,1)
        target = target.transpose(0,1)
        #print(target)
        fuse = self.positional_encoding(fuse)
        target = self.positional_encoding(target)

        #print(x.shape)
        #print(target.shape)
        out = self.transformer(src=fuse,tgt=target,
                               tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask
                               )
        #print(out.shape)

        out = self.fc(out)

        return out

# class TwoStream_Transformer(nn.Module):
        

#     def __init__(self):
#         super(TwoStream_Transformer, self).__init__()
#         d_model = 258+512  # Embedding Size
#         d_ff = 2048  # FeedForward dimension
#         n_layers = 6  # number of Encoder of Decoder Layer
#         n_heads = 10  # number of heads in Multi-Head Attention
#         self.transformer = nn.Transformer(d_model = d_model, nhead= n_heads,
#                                           num_encoder_layers=n_layers, num_decoder_layers=n_layers,
#                                           dim_feedforward=d_ff)
#         self.positional_encoding = PositionalEncoding(258+512, dropout=0)
#         self.tgt_embedding = nn.Embedding(507, 258+512, padding_idx=2)

#         self.fc = nn.Linear(258+512, 507)
        
#         resnet = models.resnet18(pretrained=True)
#         modules = list(resnet.children())[:-1]
#         self.resnet = nn.Sequential(*modules)
        
#     def get_key_padding_mask(tokens):
#         key_padding_mask = torch.zeros(tokens.size())
#         key_padding_mask[tokens == 2] = -torch.inf
#         return key_padding_mask

#     def forward(self,images, sequence,target,teacher_forcing_ratio=0.5):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         images = images.to(device)
#         sequence = sequence.to(device)
#         cnn_embed_seq = []
#         # x: (batch_size, channel, t, h, w)
#         for t in range(images.size(2)):
#             # with torch.no_grad():
#             out = self.resnet(images[:, :, t, :, :])
#             # print(out.shape)
#             out = out.view(out.size(0), -1)
#             cnn_embed_seq.append(out)

#         cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).to(device)
#         # batch first
#         cnn_embed_seq = cnn_embed_seq.transpose_(0, 1)

#         print("cnn",cnn_embed_seq.shape)

#         fuse = torch.cat((sequence, cnn_embed_seq), dim=2).to(device)
#         print("fuse",fuse.shape)


#         tgt_key_padding_mask = target.data.eq(2).cuda()

#         target = self.tgt_embedding(target)
#         tgt_mask = self.transformer.generate_square_subsequent_mask(9).to(device)


#         fuse= fuse.transpose(0,1)
#         target = target.transpose(0,1)
#         #print(target)
#         fuse = self.positional_encoding(fuse)
#         target = self.positional_encoding(target)

#         #print(x.shape)
#         print(target.shape)
#         out = self.transformer(src=fuse,tgt=target,
#                                tgt_mask=tgt_mask
#                                )
#         print(out.shape)

#         out = self.fc(out)

#         return out

class CSL_Daily_TwoStream_Transformer(nn.Module):
        
    '''
    def __init__(self):
        super(CSL_Daily_TwoStream_Transformer, self).__init__()
        d_model = 258+512  # Embedding Size
        d_ff = 4096  # FeedForward dimension
        n_layers = 24  # number of Encoder of Decoder Layer
        n_heads = 10  # number of heads in Multi-Head Attention
        self.transformer = nn.Transformer(d_model = d_model, nhead= n_heads,
                                          num_encoder_layers=n_layers, num_decoder_layers=n_layers,
                                          dim_feedforward=d_ff)
        self.positional_encoding = PositionalEncoding(258+512, dropout=0)
        self.tgt_embedding = nn.Embedding(7750+3, 258+512, padding_idx=0)

        #self.fc = nn.Linear(258+512, 7750+3)
        self.fc1 = nn.Linear(258+512, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 7750+3)
        
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
    '''
    def __init__(self):
        super(CSL_Daily_TwoStream_Transformer, self).__init__()
        d_model = 512  # Embedding Size
        d_ff = 2048  # FeedForward dimension
        n_layers = 3  # number of Encoder of Decoder Layer
        n_heads = 8 # number of heads in Multi-Head Attention
        self.transformer = nn.Transformer(d_model = d_model, nhead= n_heads,
                                          num_encoder_layers=n_layers, num_decoder_layers=n_layers,
                                          dim_feedforward=d_ff)
        self.positional_encoding = PositionalEncoding(512, dropout=0.1)
        self.tgt_embedding = nn.Embedding(7750+3, 512, padding_idx=0)

        self.fc = nn.Linear(512, 7750+3)
        self.dropout = nn.Dropout(p=0.1)
        #self.fc1 = nn.Linear(512, 2048)
        #self.fc2 = nn.Linear(2048, 4096)
        #self.fc3 = nn.Linear(4096, 7750+3)
        #self.relu = torch.nn.ReLU()
        # resnet = models.resnet34(pretrained=True)
        # modules = list(resnet.children())[:-1]
        # self.resnet1 = nn.Sequential(*modules)
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet2 = nn.Sequential(*modules)
        #CSL_Daily_TwoStream_Transformer.freeze(self.resnet2)

    def freeze(layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False

    def get_key_padding_mask(tokens):
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask

    def forward(self,images, sequence,target,teacher_forcing_ratio=0.5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tgt_mask = self.transformer.generate_square_subsequent_mask(target.size(-1)).to(device)



        images = images.to(device)
        # sequence = sequence.to(device)
        # cnn_embed_seq1 = []
        # # x: (batch_size, channel, t, h, w)
        # for t in range(images.size(2)):
        #     # with torch.no_grad():
        #     out = self.resnet1(images[:, :, t, :, :])
        #     # print(out.shape)
        #     out = out.view(out.size(0), -1)
        #     cnn_embed_seq1.append(out)

        # cnn_embed_seq1 = torch.stack(cnn_embed_seq1, dim=0).to(device)
        # # batch first
        # cnn_embed_seq1 = cnn_embed_seq1.transpose_(0, 1)
        


        cnn_embed_seq2 = []
        # x: (batch_size, channel, t, h, w)
        for t in range(images.size(2)):
            # with torch.no_grad():
            out = self.resnet2(images[:, :, t, :, :])
            # print(out.shape)
            out = out.view(out.size(0), -1)
            cnn_embed_seq2.append(out)

        cnn_embed_seq2 = torch.stack(cnn_embed_seq2, dim=0).to(device)
        # batch first
        cnn_embed_seq2 = cnn_embed_seq2.transpose_(0, 1)
        #print("cnn",cnn_embed_seq.shape)

        #fuse = torch.cat((sequence, cnn_embed_seq2), dim=2).to(device)
        # fuse = sequence
        #print("fuse",fuse.shape)
        #fuse = cnn_embed_seq
        #fuse = torch.cat((cnn_embed_seq1,sequence, cnn_embed_seq2), dim=2).to(device)
        #fuse = torch.cat((cnn_embed_seq1,cnn_embed_seq2), dim=2).to(device)
        fuse = cnn_embed_seq2

        tgt_key_padding_mask = target.data.eq(0).cuda()

        target = self.tgt_embedding(target)
        

        fuse= fuse.transpose(0,1)
        target = target.transpose(0,1)
        #print(target)
        fuse = self.positional_encoding(fuse)
        target = self.positional_encoding(target)

        #print(x.shape)
        #print(target.shape)
        out = self.transformer(src=fuse,tgt=target,
                               tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask
                               )
        print("out",out.shape)
        out = self.dropout(out)
        out = self.fc(out)
        # out = self.fc1(out)
        # out = self.relu(out)
        # out = self.dropout(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        # out = self.dropout(out)
        # out = self.fc3(out)

        return out

# Test
if __name__ == '__main__':
    # # test encoder
    # encoder = Encoder(lstm_hidden_size=512)
    # # imgs = torch.randn(16, 3, 8, 128, 128)
    # # print(encoder(imgs))

    # # test decoder
    # decoder = Decoder(output_dim=500, emb_dim=256, enc_hid_dim=512, dec_hid_dim=512, dropout=0.5)

    # # test seq2seq
    # device = torch.device("cpu")
    # seq2seq = Seq2Seq(encoder=encoder, decoder=decoder, device=device)
    # imgs = torch.randn(16, 3, 8, 128, 128)
    # target = torch.LongTensor(16, 8).random_(0, 500)
    # print(seq2seq(imgs, target).argmax(dim=2).permute(1,0)) # batch first
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSL_Daily_TwoStream_Transformer().to(device)
    images = torch.randn(1, 3, 48, 1280, 720).to(device)
    sequence = torch.randn(1, 48 ,258).to(device)
    target = torch.randint(0, 9, (1, 9)).to(device)
    out = model(images,sequence,target)
    print("out ",out.shape)
