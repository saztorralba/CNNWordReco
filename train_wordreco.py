import sys
import pandas as pd
import numpy as np
import os
import random
import argparse

from utils.cnn_func import load_data, train_model, validate_model
from models.SimpleCNN import SimpleCNN

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')

def make_folder_for_file(fileName):
    folder = os.path.dirname(fileName)
    if folder != '' and not os.path.isdir(folder):
        os.makedirs(folder)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train an isolated spoken word recogniser with CNNs')
    parser.add_argument('--input_file', metavar='FILE', default=None, help='Full path to a CSV file detailing the training data')
    parser.add_argument('--output_file', metavar='FILE', default=None, help='Full path to the output model file as a torch serialised object')
    parser.add_argument('--cv_percentage',default=0.1,type=float,help='Amount of data to use for cross-validation')
    parser.add_argument('--xsize',default=40,type=int,help='Input image size in the x axis')
    parser.add_argument('--ysize',default=40,type=int,help='Input image size in the y axis')
    parser.add_argument('--num_blocks',default=2,type=int,help='Number of convolutional blocks')
    parser.add_argument('--channels',default=8,type=int,help='Number of output convolution channels')
    parser.add_argument('--dropout',default=0.2,type=float,help='Probability of drop out for each channel')
    parser.add_argument('--embedding_size',default=64,type=int,help='Size of the intermediate embedding layer')
    parser.add_argument('--epochs',type=int,default=10,help='Number of epochs to train')
    parser.add_argument('--batch_size',type=int,default=32,help='Batch size')
    parser.add_argument('--learning_rate',type=float,default=0.1,help='Learning rate')
    parser.add_argument('--augment',default=False,action='store_true',help='Do augmentation on training data')
    parser.add_argument('--seed',type=float,default=0,help='Random seed')
    parser.add_argument('--verbose',default=0,type=int,choices=[0,1,2],help='Verbosity level (0, 1 or 2)')
    args = parser.parse_args()
    args = vars(args)
    return args

def train_wordreco(args):
    #Initialisations
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    args['device'] = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    #Load data
    data = pd.read_csv(args['input_file'])
    args['vocab'] = {w:i for i,w in enumerate(np.unique(data['word']))} 
    trainset, validset, trainlabels, validlabels = load_data(data,True,**args)
    args['mean'] = torch.mean(trainset.float())
    args['std'] = torch.std(trainset.float())
    if args['verbose'] >= 1:
        print('Number of training samples: {0:d}'.format(trainset.shape[0]))
        print('Number of cross-validaton samples: {0:d}'.format(validset.shape[0]))

    #Create model, optimiser and criterion
    model = SimpleCNN(**args).to(args['device']) 
    optimizer = torch.optim.Adam(model.parameters(),lr=args['learning_rate'])
    criterion = nn.NLLLoss(reduction='mean').to(args['device'])
    if args['verbose'] == 2:
        print('\nModel:')
        print(model)
        print('\n')

    #Train epochs
    for ep in range(1,args['epochs']+1):
        if args['verbose'] == 2:
            print('Epoch {0:d} of {1:d}'.format(ep,args['epochs']))
        loss = train_model(trainset,trainlabels,model,optimizer,criterion,**args)
        acc = validate_model(validset,validlabels,model,**args)
        if args['verbose'] == 1:
            print('Epoch {0:d} of {1:d}. Training loss: {2:.2f}, cross-validation accuracy: {3:.2f}%'.format(ep,args['epochs'],loss,acc))

        #Save intermediate models
        nfolder = os.path.dirname(args['output_file'])
        nfile = nfolder+'/intermediate/model_epoch{0:02d}_acc{1:0.2f}.pytorch'.format(ep,acc)
        make_folder_for_file(nfile)
        torch.save(model,nfile)
    if args['verbose'] == 0:
        print('Final training loss: {0:.2f}, cross-validation accuracy: {1:.2f}%'.format(loss,acc))

    #Save final models
    model = model.cpu().eval()
    make_folder_for_file(args['output_file'])
    torch.save(model,args['output_file'])

if __name__ == '__main__':
    args = parse_arguments()
    train_wordreco(args)

