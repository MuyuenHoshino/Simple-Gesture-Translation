import torch
from sklearn.metrics import accuracy_score
from tools import wer
from nltk.translate.bleu_score import sentence_bleu

# 根据字典的值value获得该值对应的key
def get_dict_key(dic, value):
    key = list(dic.keys())[list(dic.values()).index(value)]
    return key



def train_seq2seq_fuse_s3d(model, criterion, optimizer, clip, dataloader, device, epoch, logger, log_interval, writer,total_dict):
    model.train()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    for batch_idx, (imgs, sequence, target) in enumerate(dataloader):
        imgs = imgs.to(device)
        target = target.to(device)
        target_old = target
        optimizer.zero_grad()
        # forward
        a , outputs = model(imgs)
        print("output.shape",outputs.shape)
        # target: (batch_size, trg len)
        # outputs: (trg_len, batch_size, output_dim)
        # skip sos

        prediction_dim = outputs.shape[-1]
        prediction = outputs[0:].reshape([-1,prediction_dim])
        target = target.permute(1,0)[0:].reshape(-1)

        # compute the accuracy
        prediction = torch.max(prediction, 1)[1]
        #score = accuracy_score(target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())
        all_trg.extend(target)
        all_pred.extend(prediction)

        # compute wer
        # prediction: ((trg_len-1)*batch_size)
        # target: ((trg_len-1)*batch_size)
        batch_size = imgs.shape[0]
        prediction = prediction.view(-1, batch_size).permute(1,0).tolist()
        target = target.view(-1, batch_size).permute(1,0).tolist()
        wers = []
        for i in range(batch_size):
            # add mask(remove padding, sos, eos)
            print("target",target[i])
            print("prediction",prediction[i])
            print(" ")
            prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
            target[i] = [item for item in target[i] if item not in [0,1,2]]
            
            # temp1 = [[get_dict_key(total_dict,item) for item in target[i]]]
            
            # temp2 = [get_dict_key(total_dict,item) for item in prediction[i]]

            # print(" ",temp1)
            # print(temp2," ")


            wers.append(wer(target[i], prediction[i]))
        all_wer.extend(wers)



        input_length = torch.tensor(32).to(device)
        input_length = input_length.repeat(2)
        target_length = torch.tensor([len(target[0]),len(target[1])]).to(device)
        
        # compute the loss
        loss = criterion(outputs, target_old,input_length,target_length)
        losses.append(loss.item())

        # backward & optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f} | WER {:.2f}%".format(epoch+1, batch_idx+1, loss.item(), sum(wers)/len(wers)))

    # Compute the average loss & accuracy
    training_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    #training_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    training_wer = sum(all_wer)/len(all_wer)
    # Log
    writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'train': training_acc}, epoch+1)
    writer.add_scalars('WER', {'train': training_wer}, epoch+1)
    logger.info("Average Training Loss of Epoch {}: {:.6f} | WER {:.2f}%".format(epoch+1, training_loss, training_wer))
