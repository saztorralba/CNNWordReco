import sys
import os
import math
import pandas as pd
import numpy as np
import argparse

from utils.cnn_func import load_data, test_model
import torch

import warnings
warnings.filterwarnings('ignore')

def show_matrix(conf_matrix, **kwargs):
    words = list(kwargs['vocab'].keys())
    width = 1 + max([len(w) for w in words])
    print('Confusion matrix: \n')
    out='|'+'|'.rjust(width)
    for w in words:
        out+=(w+'|').rjust(width)
    print(out)
    out='|'+''.join(['-' for i in range(width-1)])+'|'
    for w in words:
        out+=''.join(['-' for i in range(width-1)])+'|'
    print(out)
    for w in words:
        out='|'+(w+'|').rjust(width)
        for w2 in words:
            out+=('{0:d}|'.format(int(conf_matrix[kwargs['vocab'][w]][kwargs['vocab'][w2]]))).rjust(width)
        print(out)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test an isolated spoken word recogniser with CNNs')
    parser.add_argument('--input_file', metavar='FILE', default=None, help='Full path to a file containing normalised sentences')
    parser.add_argument('--model_file', metavar='FILE', default=None, help='Full path to trained serialised model')
    parser.add_argument('--batch_size',type=int,default=64,help='Batch size')
    parser.add_argument('--conf_matrix',default=False,action='store_true',help='Show the confusion matrix')
    parser.add_argument('--use_tqdm',default=False,action='store_true',help='Use tqdm progress bar')
    args = parser.parse_args()
    args = vars(args)
    return args

def test_wordreco(args):
    #Read model
    model = torch.load(args['model_file']).cpu().eval()
    args['xsize'] = model.xsize
    args['ysize'] = model.ysize
    args['vocab'] = model.vocab

    #Load data
    data = pd.read_csv(args['input_file'])
    testset, testlabels = load_data(data, False, **args)

    #Compute results
    conf_matrix = test_model(testset,testlabels,model,**args)
    print('Global accuracy: {0:.2f}%'.format(100*np.sum(conf_matrix*np.eye(len(args['vocab'])))/np.sum(conf_matrix)))

    #Show confusion matrix
    if args['conf_matrix']:
        show_matrix(conf_matrix, **args)

if __name__ == '__main__':
    args=parse_arguments()
    test_wordreco(args)

