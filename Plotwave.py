import torch
import warnings
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy import signal
from scipy.interpolate import make_interp_spline
def smooth_xy(lx, ly):

    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 50000)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return [x_smooth, y_smooth]

def displayWaveform(noisy_path,speech_path,mse_path,clean04_path,clean07_path,):
    """
    display waveform of a given speech sample
    :param sample_name: speech sample name
    :param fs: sample frequency
    :return:
    """
    samples_noisy, sr = librosa.load(noisy_path, sr=16000)
    samples_noisy = samples_noisy[20000:40000]
    samples_speech, sr = librosa.load(speech_path, sr=16000)
    samples_speech = samples_speech[20000:40000]
    samples_mse, sr = librosa.load(mse_path ,sr=16000)
    samples_mse = samples_mse[20000:40000]
    samples_clean04, sr = librosa.load(clean04_path, sr=16000)
    samples_clean04 = samples_clean04[20000:40000]
    samples_clean07, sr = librosa.load(clean07_path, sr=16000)
    # print(len(samples), sr)
    time = np.arange(0, len(samples_noisy))
    peak_indexes = signal.argrelextrema(np.abs(samples_noisy), np.greater, order=500)
    peak_indexes = peak_indexes[0]
    peak_x = peak_indexes
    peak_y = abs(samples_noisy[peak_indexes])
    xy_s = smooth_xy(peak_x ,peak_y)
    plt.plot(xy_s[0],xy_s[1], color='red', label='Noisy')

    peak_indexes = signal.argrelextrema(np.abs(samples_speech), np.greater, order=500)
    peak_indexes = peak_indexes[0]
    peak_x = peak_indexes
    peak_y = abs(samples_speech[peak_indexes])
    xy_s = smooth_xy(peak_x ,peak_y)
    plt.plot(xy_s[0],xy_s[1], color='green', label='clean')

    peak_indexes = signal.argrelextrema(np.abs(samples_mse), np.greater, order=500)
    peak_indexes = peak_indexes[0]
    peak_x = peak_indexes
    peak_y = abs(samples_mse[peak_indexes])
    xy_s = smooth_xy(peak_x ,peak_y)
    plt.plot(xy_s[0],xy_s[1], color='blue', label='mse')

    # peak_indexes = signal.argrelextrema(np.abs(samples_clean04), np.greater, order=500)
    # peak_indexes = peak_indexes[0]
    # peak_x = peak_indexes
    # peak_y = abs(samples_clean04[peak_indexes])
    # xy_s = smooth_xy(peak_x ,peak_y)
    # plt.plot(xy_s[0],xy_s[1], color='black', label='CL')
    # plt.plot(time,np.square(np.abs(samples_noisy[0:40000])), color='red',label='Noisy')
    # plt.plot(time,np.square(np.abs(samples_speech[0:40000])), color='green',label='clean')
    # plt.plot(time,np.square(np.abs(samples_mse[0:56000])), color='blue',label='Mse')
    # plt.plot(samples_clean04[0:500], color='deepskyblue',label='0.4')
    # plt.plot(samples_clean07[0:500], color='gray',label='0.7')
    plt.title("Waveform comparison")
    plt.xlabel("time")
    plt.ylabel("Amplitude")
    # plt.savefig("your dir", dpi=600)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    wav_file1 = "./output_data/Noisy/D4_758_1_-5db_factory.wav"
    wav_file2 = "./output_data/Speech/D4_758_1_-5db_factory.wav"
    wav_file3 = "./output_data/STAGE1_clean_1.0/D4_758_1_-5db_factory.wav"
    wav_file4 = "./output_data/STAGE1_clean_0.4/D4_758_1_-5db_factory.wav"
    wav_file5 = "./output_data/STAGE1_clean_0.7/D4_758_1_-5db_factory.wav"

    displayWaveform(wav_file1,wav_file2,wav_file3,wav_file4,wav_file5)



    print("OK")