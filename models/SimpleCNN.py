import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, **kwargs):
        super(SimpleCNN, self).__init__()
        #Arguments
        self.xsize = kwargs['xsize']
        self.ysize = kwargs['ysize']
        self.num_blocks = kwargs['num_blocks']
        self.channels = kwargs['channels']
        self.input_channels = (kwargs['input_channels'] if 'input_channels' in kwargs else 1)
        self.reduce_size = (kwargs['reduce_size'] if 'reduce_size' in kwargs else False)
        self.dropout = kwargs['dropout']
        self.embedding_size = kwargs['embedding_size']
        self.vocab = kwargs['vocab']
        self.num_classes = len(self.vocab)
        self.mean = kwargs['mean']
        self.std = kwargs['std']

        #Gaussian normalise the input
        self.inputnorm = InputNorm(self.mean,self.std)
        #Residual convolutional block
        self.convblock1 = ConvBlock(kwargs['input_channels'], self.channels, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, dropout=self.dropout, residual=False)
        tmp_xsize = int(self.xsize/2)
        tmp_ysize = int(self.ysize/2)
        for i in range(1,self.num_blocks):
            setattr(self,'convblock'+str(i+1),ConvBlock(self.channels, self.channels, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=int(self.channels/2), dropout=self.dropout, residual=True))
            if self.reduce_size:
                setattr(self,'convblock'+str(i+1)+'b',ConvBlock(self.channels, self.channels, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=int(self.channels/2), dropout=self.dropout, residual=False))
                tmp_xsize = int(self.xsize/2)
                tmp_ysize = int(self.ysize/2)
        #Flatten the output
        self.flatten = nn.Flatten()
        #Reduce to embedding layer
        self.linear = nn.Linear(int(tmp_xsize*tmp_ysize*self.channels), self.embedding_size, bias=False)
        #Batch normalise
        self.batchnorm = nn.BatchNorm1d(self.embedding_size, momentum=0.9)
        #L2 normalise
        self.l2norm = L2Norm()
        #Classification layer and softmax
        self.output = nn.Linear(self.embedding_size, self.num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.inputnorm(x)
        out = self.convblock1(out)
        for i in range(1,self.num_blocks):
            conv = getattr(self,'convblock'+str(i+1))
            out = conv(out)
            if self.reduce_size:
                conv = getattr(self,'convblock'+str(i+1)+'b')
                out = conv(out)
        out = self.flatten(out)
        out = self.linear(out)
        out = self.batchnorm(out)
        out = self.l2norm(out)
        out = self.output(out)
        out = self.softmax(out)
        return out

#Performs gaussian normalisation of an input with mean and standard deviation
class InputNorm(nn.Module):
    def __init__(self, mean, std):
        super(InputNorm, self).__init__()
        self.mean = mean
        self.std = std
    def forward(self,x):
        out = torch.mul(torch.add(x,-self.mean),1/self.std)
        return out

#Residual convolutional block
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, dropout=0.2, residual=False):
        super(ConvBlock, self).__init__()
        #2D convolution
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        #Batch normalisation
        self.bn = nn.BatchNorm2d(out_c, momentum=0.9)
        #Activation
        self.prelu = nn.PReLU(out_c)
        #Dropout
        self.dropout = nn.Dropout3d(p=dropout)
        self.residual = residual
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.dropout(out)
        #Residual connection
        if self.residual:
            out = out + x
        return out

#Do L2 normalisation of embedding vectors
class L2Norm(nn.Module):
    def __init__(self, axis=1):
        super(L2Norm, self).__init__()
        self.axis = axis
    def forward(self,x):
        norm = torch.norm(x, 2, self.axis, True)
        output = torch.div(x, norm)
        return output
