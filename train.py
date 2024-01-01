import torch
from sklearn.metrics import accuracy_score
from tools import wer
from nltk.translate.bleu_score import sentence_bleu

# 根据字典的值value获得该值对应的key
def get_dict_key(dic, value):
    key = list(dic.keys())[list(dic.values()).index(value)]
    return key


def train_epoch(model, criterion, optimizer, dataloader, device, epoch, logger, log_interval, writer):
    model.train()
    losses = []
    all_label = []
    all_pred = []

    for batch_idx, data in enumerate(dataloader):
        # get the inputs and labels
        inputs, labels = data['data'].to(device), data['label'].to(device)

        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        if isinstance(outputs, list):
            outputs = outputs[0]

        # compute the loss
        loss = criterion(outputs, labels.squeeze())
        losses.append(loss.item())

        # compute the accuracy
        prediction = torch.max(outputs, 1)[1]
        all_label.extend(labels.squeeze())
        all_pred.extend(prediction)
        score = accuracy_score(labels.squeeze().cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())

        # backward & optimize
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}%".format(epoch+1, batch_idx+1, loss.item(), score*100))

    # Compute the average loss & accuracy
    training_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    # Log
    writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'train': training_acc}, epoch+1)
    logger.info("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch+1, training_loss, training_acc*100))


def train_seq2seq(model, criterion, optimizer, clip, dataloader, device, epoch, logger, log_interval, writer):
    model.train()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []


    for batch_idx, (imgs, target) in enumerate(dataloader):
        imgs = imgs.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        # forward
        outputs = model(imgs, target)

        # target: (batch_size, trg len)
        # outputs: (trg_len, batch_size, output_dim)
        # skip sos
        output_dim = outputs.shape[-1]
        outputs = outputs[1:].view(-1, output_dim)
        target = target.permute(1,0)[1:].reshape(-1)

        # compute the loss
        loss = criterion(outputs, target)
        losses.append(loss.item())

        # compute the accuracy
        prediction = torch.max(outputs, 1)[1]
        score = accuracy_score(target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())
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
            prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
            target[i] = [item for item in target[i] if item not in [0,1,2]]
            wers.append(wer(target[i], prediction[i]))
        all_wer.extend(wers)

        # backward & optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}% | WER {:.2f}%".format(epoch+1, batch_idx+1, loss.item(), score*100, sum(wers)/len(wers)))

    # Compute the average loss & accuracy
    training_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    training_wer = sum(all_wer)/len(all_wer)
    # Log
    writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'train': training_acc}, epoch+1)
    writer.add_scalars('WER', {'train': training_wer}, epoch+1)
    logger.info("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER {:.2f}%".format(epoch+1, training_loss, training_acc*100, training_wer))



def train_seq2seq_fuse(model, criterion, optimizer, clip, dataloader, device, epoch, logger, log_interval, writer, total_dict):
    model.train()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    all_bleu4 = []

    for batch_idx, (imgs, sequence, target) in enumerate(dataloader):
        imgs = imgs.to(device)
        target = target.to(device)
        print("target",target)
        optimizer.zero_grad()
        # forward
        outputs = model(imgs, sequence ,target)
        #print("output",outputs)
        # target: (batch_size, trg len)
        # outputs: (trg_len, batch_size, output_dim)
        # skip sos
        output_dim = outputs.shape[-1]
        outputs = outputs[0:-1].view(-1, output_dim)
        target = target.permute(1,0)[1:].reshape(-1)

        print("target for loss",target.shape)
        print("target for loss",target)        
        print("outputs for loss",outputs.shape)
        print("outputs for loss",outputs)
        # compute the loss


        #loss = criterion(outputs[0:10], target[0:10])
        loss = criterion(outputs, target)
        losses.append(loss.item())

        # compute the accuracy
        prediction = torch.max(outputs, 1)[1]
        score = accuracy_score(target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())
        all_trg.extend(target)
        all_pred.extend(prediction)

        # compute wer
        # prediction: ((trg_len-1)*batch_size)
        # target: ((trg_len-1)*batch_size)
        batch_size = imgs.shape[0]
        prediction = prediction.view(-1, batch_size).permute(1,0).tolist()
        target = target.view(-1, batch_size).permute(1,0).tolist()
        wers = []

        bleu4 = []

        for i in range(batch_size):
            # add mask(remove padding, sos, eos)
            prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
            target[i] = [item for item in target[i] if item not in [0,1,2]]
            wers.append(wer(target[i], prediction[i]))

            temp1 = [[get_dict_key(total_dict,item) for item in target[i]]]
            
            temp2 = [get_dict_key(total_dict,item) for item in prediction[i]]

            print(temp1)
            print(temp2)

            #temp = sentence_bleu(target[i], prediction[i], weights=(0, 0, 0, 1))
            temp = sentence_bleu(temp1, temp2, weights=(0, 0, 0, 1))
            bleu4.append(temp)
            print(bleu4)

        all_wer.extend(wers)
        all_bleu4.extend(bleu4)

        # backward & optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        #print("BLEU-4:---------------",bleu4)
        if (batch_idx + 1) % log_interval == 0:
            logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}% | WER {:.2f}% | BLEU-4 {:.2f}".format(epoch+1, batch_idx+1, loss.item(), score*100, sum(wers)/len(wers), 100*sum(bleu4)/len(bleu4)))

    # Compute the average loss & accuracy
    training_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    training_wer = sum(all_wer)/len(all_wer)

    training_bleu4 = sum(all_bleu4)/len(all_bleu4)

    # Log
    writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'train': training_acc}, epoch+1)
    writer.add_scalars('WER', {'train': training_wer}, epoch+1)
    writer.add_scalars('BLEU-4', {'train': training_bleu4}, epoch+1)
    logger.info("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER {:.2f}% | BLEU-4 {:.2f}".format(epoch+1, training_loss, training_acc*100, training_wer, training_bleu4*100))





def train_seq2seq_fuse_old(model, criterion, optimizer, clip, dataloader, device, epoch, logger, log_interval, writer,total_dict):
    model.train()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    for batch_idx, (imgs, sequence, target) in enumerate(dataloader):
        imgs = imgs.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        # forward
        outputs = model(imgs,sequence ,target)

        # target: (batch_size, trg len)
        # outputs: (trg_len, batch_size, output_dim)
        # skip sos
        output_dim = outputs.shape[-1]
        outputs = outputs[0:-1].view(-1, output_dim)
        target = target.permute(1,0)[1:].reshape(-1)

        # compute the loss
        loss = criterion(outputs, target)
        losses.append(loss.item())

        # compute the accuracy
        prediction = torch.max(outputs, 1)[1]
        score = accuracy_score(target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())
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

        # backward & optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}% | WER {:.2f}%".format(epoch+1, batch_idx+1, loss.item(), score*100, sum(wers)/len(wers)))

    # Compute the average loss & accuracy
    training_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    training_wer = sum(all_wer)/len(all_wer)
    # Log
    writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'train': training_acc}, epoch+1)
    writer.add_scalars('WER', {'train': training_wer}, epoch+1)
    logger.info("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER {:.2f}%".format(epoch+1, training_loss, training_acc*100, training_wer))


def train_seq2seq_fuse_greedy(model, criterion, optimizer, clip, dataloader, device, epoch, logger, log_interval, writer,total_dict):
    model.train()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    all_bleu4 = []


    for batch_idx, (imgs, sequence,target) in enumerate(dataloader):
        imgs = imgs.to(device)
        target = target.to(device)
        
        batch_size = imgs.shape[0]
        
        tensor_list = []
        for i in range(batch_size):
            tensor_list.append([1])
        greed_decode = torch.LongTensor(tensor_list)
        greed_decode = greed_decode.to(device)

        outputs = []
        
        for i in range(31-1):
            #print(imgs.shape)
            
            print("greed_decode",greed_decode.shape)

            model_outputs = model(imgs,sequence ,greed_decode)
            #model_outputs = model_outputs.to(device)
            print("model_outputs",model_outputs.shape)

            predict = model_outputs[-1,:]
            #predict = predict.to(device)
            print("predict",predict.shape)

            outputs.append(predict)

            greed = torch.argmax(predict, dim=1)
            #greed = greed.to(device)
            print("greed",greed.shape)
            
            greed_decode = torch.cat((greed_decode, greed.view(batch_size,-1)), dim=1)
            #greed_decode = greed_decode.to(device)
            print("greed_decode",greed_decode.shape)

        outputs = torch.cat(outputs, dim=0)
        print("outputs for loss",outputs.shape)
        print("outputs for loss",outputs)
        # forward(no teacher forcing)
        #outputs = model(imgs,sequence ,target, 0)

        # target: (batch_size, trg len)
        # outputs: (trg_len, batch_size, output_dim)
        # skip sos


        output_dim = outputs.shape[-1]
        #outputs = outputs[1:].view(-1, output_dim)
        outputs = outputs.view(-1, output_dim)
        #这里已经去掉第一项了
        target = target.permute(1,0)[1:].reshape(-1)

        print("target for loss",target.shape)
        print("target for loss",target)
        print("greed_decode",greed_decode.shape)
        print("greed_decode",greed_decode)
        # compute the loss
        loss = criterion(outputs, target)
        losses.append(loss.item())

        # compute the accuracy
        prediction = torch.max(outputs, 1)[1]
        print("greed_decode",greed_decode.shape)
        print("greed_decode",greed_decode)
        score = accuracy_score(target.cpu().data.squeeze().numpy(), prediction.cpu().data.squeeze().numpy())
        all_trg.extend(target)
        all_pred.extend(prediction)

        # compute wer
        # prediction: ((trg_len-1)*batch_size)
        # target: ((trg_len-1)*batch_size)
        batch_size = imgs.shape[0]
        prediction = prediction.view(-1, batch_size).permute(1,0).tolist()
        target = target.view(-1, batch_size).permute(1,0).tolist()
        wers = []

        bleu4 = []
        for i in range(batch_size):
            # add mask(remove padding, eos, sos)
            prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
            target[i] = [item for item in target[i] if item not in [0,1,2]]
            wers.append(wer(target[i], prediction[i]))

            print(prediction[i])
            print(target[i])
            temp1 = [[get_dict_key(total_dict,item) for item in target[i]]]
            temp2 = [get_dict_key(total_dict,item) for item in prediction[i]]

            #temp = sentence_bleu(target[i], prediction[i], weights=(0, 0, 0, 1))
            temp = sentence_bleu(temp1, temp2, weights=(0, 0, 0, 1))
            bleu4.append(temp)
            print(bleu4)

        all_wer.extend(wers)
        all_bleu4.extend(bleu4)

        # backward & optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            logger.info("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}% | WER {:.2f}%".format(epoch+1, batch_idx+1, loss.item(), score*100, sum(wers)/len(wers)))

    # Compute the average loss & accuracy
    training_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    training_wer = sum(all_wer)/len(all_wer)
    # Log
    writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'train': training_acc}, epoch+1)
    writer.add_scalars('WER', {'train': training_wer}, epoch+1)
    logger.info("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER {:.2f}%".format(epoch+1, training_loss, training_acc*100, training_wer))


