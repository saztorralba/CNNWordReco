import math
import numpy as np
import soundfile
from librosa import resample
import librosa.feature
import scipy.signal
from PIL import Image

#Function to extract log-mel spectrums
def gen_logmel(signal,n_mels,fs=8000,normalise=False,n_fft=25,hop_length=10,f=None):
    epsilon=1e-20
    n_fft=int(n_fft*fs/1000)
    hop_length=int(hop_length*fs/1000)
    #Read file
    if isinstance(signal,str):
        audio,f=soundfile.read(signal)
        if f!=fs:
            audio=resample(audio,f,fs)
    else:
        if f is not None and f!=fs:
            audio=resample(signal,f,fs)
    #Normalise input energy
    if normalise:
        norm = np.max(np.absolute(audio))
        if norm > 0:
            audio=0.5*audio/norm
    #High-pass filter
    #audio=scipy.signal.convolve(audio,np.array([1,-0.98]),mode='same',method='fft')
    audio = audio[1:] - 0.98 * audio[:-1]
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
