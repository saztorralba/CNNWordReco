import math
import numpy as np
import soundfile
from librosa import resample
import librosa.feature
import scipy.signal
from PIL import Image

#Function to extract log-mel spectrums
def gen_logmel(signal,n_mels,fs=8000,normalise=False,n_fft=25,hop_length=10):
    epsilon=1e-20
    n_fft=int(n_fft*fs/1000)
    hop_length=int(hop_length*fs/1000)
    #Read file
    audio,f=soundfile.read(signal)
    if f!=fs:
        audio=resample(audio,f,fs)
    #Normalise input energy
    if normalise:
        audio=0.5*audio/np.max(np.absolute(audio))
    #High-pass filter
    audio=scipy.signal.convolve(audio,np.array([1,-0.98]),mode='same',method='fft')
    #Comput spectrogram
    melspec=librosa.feature.melspectrogram(y=audio,sr=fs,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels,fmin=100,fmax=fs/2,norm=1)
    #Logarithm
    DATA=np.transpose(np.log(melspec+epsilon))
    #Discard last incomplete frame
    DATA=DATA[0:math.floor((audio.shape[0]-(n_fft-hop_length))/hop_length),:]
    return DATA

#Function to convert log-mel spectrogram to PIL image, resize and back
def feat2img(DATA,xsize=40,ysize=40):
    #Reorg dimensions
    DATA = np.flipud(np.transpose(DATA))
    #Clamp and normalise to [0,1]
    DATA = (np.maximum(-15,np.minimum(0,DATA))+15)/15
    #Convert to PIL
    im = Image.fromarray(np.uint8(DATA*255))
    #Resize
    im = im.resize((ysize,xsize))
    #Back to numpy
    DATA = np.array(im)
    return DATA