import torch
from sklearn.metrics import accuracy_score
from tools import wer
from nltk.translate.bleu_score import sentence_bleu
def get_dict_key(dic, value):
    key = list(dic.keys())[list(dic.values()).index(value)]
    return key


def val_epoch(model, criterion, dataloader, device, epoch, logger, writer):
    model.eval()
    losses = []
    all_label = []
    all_pred = []

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # get the inputs and labels
            inputs, labels = data['data'].to(device), data['label'].to(device)
            # forward
            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
            # compute the loss
            loss = criterion(outputs, labels.squeeze())
            losses.append(loss.item())
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)
    # Compute the average loss & accuracy
    validation_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    validation_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    # Log
    writer.add_scalars('Loss', {'validation': validation_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'validation': validation_acc}, epoch+1)
    logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch+1, validation_loss, validation_acc*100))

def test_epoch(model, criterion, dataloader, device, epoch, logger, writer):
    model.eval()
    losses = []
    all_label = []
    all_pred = []

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # get the inputs and labels
            inputs, labels = data['data'].to(device), data['label'].to(device)
            # forward
            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
            # compute the loss
            loss = criterion(outputs, labels.squeeze())
            losses.append(loss.item())
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)
    # Compute the average loss & accuracy
    test_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    test_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    # Log
    writer.add_scalars('Loss', {'test': test_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'test': test_acc}, epoch+1)
    logger.info("Average test Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch+1, test_loss, test_acc*100))

def val_seq2seq(model, criterion, dataloader, device, epoch, logger, writer):
    model.eval()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    with torch.no_grad():
        for batch_idx, (imgs, target) in enumerate(dataloader):
            imgs = imgs.to(device)
            target = target.to(device)

            # forward(no teacher forcing)
            outputs = model(imgs, target, 0)

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
                # add mask(remove padding, eos, sos)
                prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
                target[i] = [item for item in target[i] if item not in [0,1,2]]
                wers.append(wer(target[i], prediction[i]))
            all_wer.extend(wers)

    # Compute the average loss & accuracy
    validation_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    validation_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    validation_wer = sum(all_wer)/len(all_wer)
    # Log
    writer.add_scalars('Loss', {'validation': validation_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'validation': validation_acc}, epoch+1)
    writer.add_scalars('WER', {'validation': validation_wer}, epoch+1)
    logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER: {:.2f}%".format(epoch+1, validation_loss, validation_acc*100, validation_wer))


def test_seq2seq(model, criterion, dataloader, device, epoch, logger, writer):
    model.eval()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    with torch.no_grad():
        for batch_idx, (imgs, target) in enumerate(dataloader):
            imgs = imgs.to(device)
            target = target.to(device)

            # forward(no teacher forcing)
            outputs = model(imgs, target, 0)

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
                # add mask(remove padding, eos, sos)
                prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
                target[i] = [item for item in target[i] if item not in [0,1,2]]
                wers.append(wer(target[i], prediction[i]))
            all_wer.extend(wers)

    # Compute the average loss & accuracy
    test_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    test_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    test_wer = sum(all_wer)/len(all_wer)
    # Log
    writer.add_scalars('Loss', {'test': test_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'test': test_acc}, epoch+1)
    writer.add_scalars('WER', {'test': test_wer}, epoch+1)
    logger.info("Average test Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER: {:.2f}%".format(epoch+1, test_loss, test_acc*100, test_wer))



def val_seq2seq_fuse(model, criterion, dataloader, device, epoch, logger, writer, total_dict):
    model.eval()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    all_bleu4 = []

    with torch.no_grad():
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
            
            for i in range(9-1):
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


    # Compute the average loss & accuracy
    validation_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    validation_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    validation_wer = sum(all_wer)/len(all_wer)
    validation_bleu4 = sum(all_bleu4)/len(all_bleu4)
    # Log
    writer.add_scalars('Loss', {'validation': validation_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'validation': validation_acc}, epoch+1)
    writer.add_scalars('WER', {'validation': validation_wer}, epoch+1)
    logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER: {:.2f}% | BLEU-4: {:.2f}".format(epoch+1, validation_loss, validation_acc*100, validation_wer, validation_bleu4*100))


def test_seq2seq_fuse(model, criterion, dataloader, device, epoch, logger, writer, total_dict):
    model.eval()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    all_bleu4 = []

    with torch.no_grad():
        for batch_idx, (imgs,sequence,target) in enumerate(dataloader):
            imgs = imgs.to(device)
            target = target.to(device)
            
            batch_size = imgs.shape[0]
            
            tensor_list = []
            for i in range(batch_size):
                tensor_list.append([1])
            greed_decode = torch.LongTensor(tensor_list)
            greed_decode = greed_decode.to(device)

            outputs = []
            
            for i in range(9-1):
                
                
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

            bleu4 = []
            for i in range(batch_size):
                # add mask(remove padding, eos, sos)
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

    # Compute the average loss & accuracy
    test_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    test_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    test_wer = sum(all_wer)/len(all_wer)

    test_bleu4 = sum(all_bleu4)/len(all_bleu4)

    # Log
    writer.add_scalars('Loss', {'test': test_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'test': test_acc}, epoch+1)
    writer.add_scalars('WER', {'test': test_wer}, epoch+1)
    logger.info("Average test Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER: {:.2f}% | BLEU-4: {:.2f}".format(epoch+1, test_loss, test_acc*100, test_wer, test_bleu4*100))


def val_seq2seq_fuse_daily(model, criterion, dataloader, device, epoch, logger, writer, total_dict):
    model.eval()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    all_bleu4 = []

    with torch.no_grad():
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
                print("target",target[i])
                print("prediction",prediction[i])
                print(" ")
                
                temp1 = [[get_dict_key(total_dict,item) for item in target[i]]]
                temp2 = [get_dict_key(total_dict,item) for item in prediction[i]]


                #temp = sentence_bleu(target[i], prediction[i], weights=(0, 0, 0, 1))
                temp = sentence_bleu(temp1, temp2, weights=(0, 0, 0, 1))
                bleu4.append(temp)
                print(bleu4)

            all_wer.extend(wers)
            all_bleu4.extend(bleu4)


    # Compute the average loss & accuracy
    validation_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    validation_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    validation_wer = sum(all_wer)/len(all_wer)
    validation_bleu4 = sum(all_bleu4)/len(all_bleu4)
    # Log
    writer.add_scalars('Loss', {'validation': validation_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'validation': validation_acc}, epoch+1)
    writer.add_scalars('WER', {'validation': validation_wer}, epoch+1)
    logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER: {:.2f}% | BLEU-4: {:.2f}".format(epoch+1, validation_loss, validation_acc*100, validation_wer, validation_bleu4*100))


def test_seq2seq_fuse_daily(model, criterion, dataloader, device, epoch, logger, writer, total_dict):
    model.eval()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    all_bleu4 = []

    with torch.no_grad():
        for batch_idx, (imgs,sequence,target) in enumerate(dataloader):
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

            bleu4 = []
            for i in range(batch_size):
                # add mask(remove padding, eos, sos)
                prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
                target[i] = [item for item in target[i] if item not in [0,1,2]]
                wers.append(wer(target[i], prediction[i]))

                print("target",target[i])
                print("prediction",prediction[i])
                print(" ")
                temp1 = [[get_dict_key(total_dict,item) for item in target[i]]]
                temp2 = [get_dict_key(total_dict,item) for item in prediction[i]]

                #print(temp1)
                #print(temp2)

                #temp = sentence_bleu(target[i], prediction[i], weights=(0, 0, 0, 1))
                temp = sentence_bleu(temp1, temp2, weights=(0, 0, 0, 1))
                bleu4.append(temp)
                print(bleu4)

            all_wer.extend(wers)
            all_bleu4.extend(bleu4)

    # Compute the average loss & accuracy
    test_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    test_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    test_wer = sum(all_wer)/len(all_wer)

    test_bleu4 = sum(all_bleu4)/len(all_bleu4)

    # Log
    writer.add_scalars('Loss', {'test': test_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'test': test_acc}, epoch+1)
    writer.add_scalars('WER', {'test': test_wer}, epoch+1)
    logger.info("Average test Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER: {:.2f}% | BLEU-4: {:.2f}".format(epoch+1, test_loss, test_acc*100, test_wer, test_bleu4*100))


def val_seq2seq_fuse_new(model, criterion, dataloader, device, epoch, logger, writer, total_dict):
    model.eval()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    all_bleu4 = []

    with torch.no_grad():
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

                print(temp1)
                print(temp2)
                temp1 = [[get_dict_key(total_dict,item) for item in target[i]]]
                temp2 = [get_dict_key(total_dict,item) for item in prediction[i]]



                #temp = sentence_bleu(target[i], prediction[i], weights=(0, 0, 0, 1))
                temp = sentence_bleu(temp1, temp2, weights=(0, 0, 0, 1))
                bleu4.append(temp)
                print(bleu4)

            all_wer.extend(wers)
            all_bleu4.extend(bleu4)


    # Compute the average loss & accuracy
    validation_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    validation_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    validation_wer = sum(all_wer)/len(all_wer)
    validation_bleu4 = sum(all_bleu4)/len(all_bleu4)
    # Log
    writer.add_scalars('Loss', {'validation': validation_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'validation': validation_acc}, epoch+1)
    writer.add_scalars('WER', {'validation': validation_wer}, epoch+1)
    logger.info("Average Validation Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER: {:.2f}% | BLEU-4: {:.2f}".format(epoch+1, validation_loss, validation_acc*100, validation_wer, validation_bleu4*100))


def test_seq2seq_fuse_new(model, criterion, dataloader, device, epoch, logger, writer, total_dict):
    model.eval()
    losses = []
    all_trg = []
    all_pred = []
    all_wer = []

    all_bleu4 = []

    with torch.no_grad():
        for batch_idx, (imgs,sequence,target) in enumerate(dataloader):
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

            bleu4 = []
            for i in range(batch_size):
                # add mask(remove padding, eos, sos)
                prediction[i] = [item for item in prediction[i] if item not in [0,1,2]]
                target[i] = [item for item in target[i] if item not in [0,1,2]]
                wers.append(wer(target[i], prediction[i]))

                
                print(temp1)
                print(temp2)

                temp1 = [[get_dict_key(total_dict,item) for item in target[i]]]
                temp2 = [get_dict_key(total_dict,item) for item in prediction[i]]

                #temp = sentence_bleu(target[i], prediction[i], weights=(0, 0, 0, 1))
                temp = sentence_bleu(temp1, temp2, weights=(0, 0, 0, 1))
                bleu4.append(temp)
                print(bleu4)

            all_wer.extend(wers)
            all_bleu4.extend(bleu4)

    # Compute the average loss & accuracy
    test_loss = sum(losses)/len(losses)
    all_trg = torch.stack(all_trg, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    test_acc = accuracy_score(all_trg.cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    test_wer = sum(all_wer)/len(all_wer)

    test_bleu4 = sum(all_bleu4)/len(all_bleu4)

    # Log
    writer.add_scalars('Loss', {'test': test_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'test': test_acc}, epoch+1)
    writer.add_scalars('WER', {'test': test_wer}, epoch+1)
    logger.info("Average test Loss of Epoch {}: {:.6f} | Acc: {:.2f}% | WER: {:.2f}% | BLEU-4: {:.2f}".format(epoch+1, test_loss, test_acc*100, test_wer, test_bleu4*100))

