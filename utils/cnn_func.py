import math
import numpy as np
import torch
import random
from utils.sigproc import gen_logmel, feat2img
from tqdm import tqdm

#Load all data in memory
def load_data(data,cv=False,**kwargs):
    n_samples = len(data)
    dataset = torch.zeros((n_samples,1,kwargs['ysize'],kwargs['xsize']),dtype=torch.uint8)
    labels = torch.zeros((n_samples),dtype=torch.uint8)
    for i in tqdm(range(n_samples),disable=(kwargs['verbose']<2)):
        path = data['wavfile'][i]
        dataset[i,0,:,:] = torch.from_numpy(feat2img(gen_logmel(path,40,8000,True),kwargs['ysize'],kwargs['xsize']))
        labels[i] = kwargs['vocab'][data['word'][i]]

    if cv == False:
        return dataset, labels

    #Do random train/validation split
    idx = [i for i in range(n_samples)]
    random.shuffle(idx)
    trainset = dataset[idx[0:int(n_samples*(1-kwargs['cv_percentage']))]]
    trainlabels = labels[idx[0:int(n_samples*(1-kwargs['cv_percentage']))]]
    validset = dataset[idx[int(n_samples*(1-kwargs['cv_percentage'])):]]
    validlabels = labels[idx[int(n_samples*(1-kwargs['cv_percentage'])):]]
    return trainset, validset, trainlabels, validlabels

#Train the model for an epoch
def train_model(trainset,trainlabels,model,optimizer,criterion,**kwargs):
    trainlen = trainset.shape[0]
    nbatches = math.ceil(trainlen/kwargs['batch_size'])
    total_loss = 0
    total_backs = 0
    with tqdm(total=nbatches,disable=(kwargs['verbose']<2)) as pbar:
        model = model.train()
        for b in range(nbatches):
            #Obtain batch
            X = trainset[b*kwargs['batch_size']:min(trainlen,(b+1)*kwargs['batch_size'])].clone().float()
            if kwargs['augment']:
                for i in range(X.shape[0]):
                    for j in range(X.shape[0]):
                        if random.random()<0.3:
                            locate=[random.randrange(0,kwargs['ysize']-2),random.randrange(0,kwargs['xsize']-2)]
                            X[locate[0]:locate[0]+3,locate[1]:locate[1]+3]=random.randint(0,255)
                X = X + 10*torch.randn(X.shape)
            X = X.to(kwargs['device'])
            Y = trainlabels[b*kwargs['batch_size']:min(trainlen,(b+1)*kwargs['batch_size'])].clone().long().to(kwargs['device'])
            #Propagate
            posteriors = model(X)
            #Backpropagate
            loss = criterion(posteriors,Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #Track loss
            if total_backs == 100:
                total_loss = total_loss*0.99+loss.detach().cpu().numpy()
            else:
                total_loss += loss.detach().cpu().numpy()
                total_backs += 1
            pbar.set_description(f'Training epoch. Loss {total_loss/(total_backs+1):.2f}')
            pbar.update()
    return total_loss/(total_backs+1)

#Validate last epoch's model
def validate_model(validset,validlabels,model,**kwargs):
    validlen = validset.shape[0]
    acc = 0
    total = 0
    nbatches = math.ceil(validlen/kwargs['batch_size'])
    with torch.no_grad():
        with tqdm(total=nbatches,disable=(kwargs['verbose']<2)) as pbar:
            model = model.eval()
            for b in range(nbatches):
                #Obtain batch
                X = validset[b*kwargs['batch_size']:min(validlen,(b+1)*kwargs['batch_size'])].clone().float().to(kwargs['device'])
                Y = validlabels[b*kwargs['batch_size']:min(validlen,(b+1)*kwargs['batch_size'])].clone().long().to(kwargs['device'])
                #Propagate
                posteriors = model(X)
                #Accumulate accuracy
                estimated = torch.argmax(posteriors,dim=1)
                acc += sum((estimated.cpu().numpy() == Y.cpu().numpy()))
                total+=Y.shape[0]
                pbar.set_description(f'Evaluating epoch. Accuracy {100*acc/total:.2f}%')
                pbar.update()
    return 100*acc/total

#Test a model
def test_model(testset,testlabels,model,**kwargs):
    testlen = testset.shape[0]
    conf_matrix = np.zeros((len(kwargs['vocab']),len(kwargs['vocab'])))
    nbatches = math.ceil(testlen/kwargs['batch_size'])
    with torch.no_grad():
        model = model.eval()
        with tqdm(total=nbatches,disable=(kwargs['verbose']<2)) as pbar:
            for b in range(nbatches):
                #Obtain batch
                X = testset[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])].clone().float()
                Y = testlabels[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])].clone().long()
                #Propagate
                posteriors = model(X)
                #Accumulate confusion matrix
                estimated = torch.argmax(posteriors,dim=1)
                for i in range(Y.shape[0]):
                    conf_matrix[Y[i],estimated[i]]+=1
                pbar.set_description('Testing')
                pbar.update()
    return conf_matrix
