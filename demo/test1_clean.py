import torch
import torch.nn as nn
import hparams as hp
from python_speech_features import mfcc, fbank, hz2mel, mel2hz,erb
from torch.autograd import Variable
import os
import numpy as np
import timeit
import torch.optim as optim
import warnings
import soundfile as sf
import librosa
from dataset_pytorch import get_mask
from STFT import wav_stft
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("create floder {}".format(path))

def test(wav_file, Model,hp,mean_Fbank,std_Fbank,mean_pow,std_pow,device,wav_stft):
    ################## FBANK
    noisy_wav, fs = sf.read(wav_file, dtype='float32')
    noisy_wav = noisy_wav.astype('float32')
    if (hp.bandmode == "fbank"):
        Fbank = fbank(noisy_wav, samplerate=hp.fs, winlen=0.032, winstep=0.016, nfilt=hp.num_filter, nfft=512, lowfreq=0,
                      highfreq=8000, preemph=0, winfunc=np.hanning)
    elif (hp.bandmode == "erb"):
        Fbank = erb(noisy_wav, samplerate=hp.fs, winlen=0.032, winstep=0.016, nfilt=hp.num_filter, nfft=512, lowfreq=0,
                      highfreq=8000, preemph=0, winfunc=np.hanning)
    if (hp.bandmode == "mfcc"):
        Fbank = mfcc(noisy_wav, samplerate=hp.fs, winlen=0.032, winstep=0.016, nfilt=hp.num_filter, nfft=512, lowfreq=0,
                      highfreq=8000, preemph=0, winfunc=np.hanning)
    Fbank = torch.from_numpy(Fbank)
    mean_erb = np.load(mean_Fbank)
    std_erb = np.load(std_Fbank)
    mean_erb = torch.unsqueeze(torch.from_numpy(mean_erb.T), dim=0)
    std_erb = torch.unsqueeze(torch.from_numpy(std_erb.T), dim=0)
    Fbank = (Fbank - mean_erb) / (std_erb)
    Fbank_Train  = Fbank.to(device)
    Fbank_Train  =torch.unsqueeze(Fbank_Train,dim=0)
    Fbank_Train = torch.unsqueeze(Fbank_Train, dim=0)
    ################## END
    noisy_wav = torch.from_numpy(noisy_wav)
    noisy_wav = noisy_wav.to(device)
    noisy_spec_r, noisy_spec_i, noisy_spec, noisy_mags, noisy_phase, noisy_lps = wav_stft.STFT(noisy_wav)
    ###### lps
    mean_pow = np.load(mean_pow)
    std_pow = np.load(std_pow)
    mean_pow = torch.from_numpy(mean_pow.T).to(device)
    std_pow = torch.from_numpy(std_pow.T).to(device)
    mean_lps = torch.unsqueeze(mean_pow, dim=1)
    std_lps = torch.unsqueeze(std_pow, dim=1)
    lps = (noisy_lps - mean_lps) / std_lps
    ########
    Model = Model.to(device)
    noisy_phase =noisy_phase.to(device)

    noisy_spec = torch.unsqueeze(noisy_spec,dim=0)
    X_Train = noisy_spec.permute(0, 3, 2, 1)

    X_Train = torch.reshape(X_Train, (hp.batch_size, X_Train.shape[1], -1, X_Train.shape[-1]))
    specs = X_Train.permute(0,3,2,1)
    Fbank_Train = torch.reshape(Fbank_Train, (hp.batch_size, Fbank_Train.shape[1], -1, Fbank_Train.shape[-1]))
    lps = torch.reshape(lps, (hp.batch_size, -1, lps.shape[0]))
    X_train_feat = X_Train[:, :, :, :128]

    Model.eval()
    with torch.no_grad():
        band_mask,full_mask, out_spec= Model(hp, Fbank_Train.float(),specs.float(),lps,device)
        ests_wav = wav_stft.iSTFT(out_spec)
        ests_wav = torch.squeeze(ests_wav,dim=0)
    return ests_wav

if __name__ == "__main__":
    hp =hp
    hp.batch_size = 1
    wav_stft = wav_stft()
    path= os.getcwd()
    warnings.filterwarnings("ignore", category=Warning)
    use_cuda = hp.use_cuda
    cuda =  use_cuda and torch.cuda.is_available()
    cuda_name = 'cuda:0'
    device = torch.device(cuda_name if cuda else "cpu")
    if cuda:
        gpu_num = torch.cuda.device_count()
        print(f' Detecting {gpu_num} GPUs available' \
              f'load and train the model on GPU {torch.cuda.current_device()}:{torch.cuda.get_device_name(0)}')
    else:
        print("cuda is not available")

    hp.epochs = hp.epochs - 1
    _model = "model_" + str(hp.epochs) + ".pth"
    model_name = os.path.join("STAGE1_clean", _model)
    out_path = hp.test_output_path1
    create_path(out_path)
    out_path_1 = os.path.join(path, out_path)
    model = torch.load(os.path.join(path, model_name), map_location=device)
    test_wav_path = hp.test_noisy_path
    test_wavs = os.listdir(test_wav_path)
    for name in test_wavs:
        # print("test mode process\n" + name)
        input =  os.path.join(test_wav_path, name)
        out_wav = test(input,model,hp,"mean_feature.npy","std_feature.npy","mean_pow.npy","std_pow.npy",device=device,wav_stft=wav_stft)
        out_wav = torch.squeeze(out_wav,dim=0)
        out = os.path.join(out_path_1, name)
        out_wav =out_wav.cpu()
        sf.write(out, out_wav, hp.fs, subtype='PCM_16')
        print(name)

























