import librosa
import soundfile as sf
import os
import shutil
from pystoi.stoi import stoi
import numpy as np
import argparse
import multiprocessing as mp
from _function import pesq_score
import json





def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
def calculate_pesq(args, process_queue, save_queque):
    save_pesq_dir = args.save_pesq_dir
    while True:
        if process_queue.qsize() % 2000 == 0:
            print("processed 2000 utterances, remain %d utterances" % process_queue.qsize())
            if process_queue.qsize() == 0:
                break
        line = process_queue.get()
        if line == None:
            break
        speech_path = line.split('+')[0]
        enh_path = line.split('+')[-1].strip()
        d = pesq_score.pesq(enh_path, speech_path)
        if d is not None:
            save = ("%.4f\t" % d + enh_path + "\n")
            save_queque.put(save)
def calculate_stoi(args, process_queue, save_queque):

    while True:
        if process_queue.qsize() % 2000 == 0:
            print("processed 2000 utterances, remain %d utterances" % process_queue.qsize())
            if process_queue.qsize() == 0:
                break
        line = process_queue.get()
        if line == None:
            break
        speech_path = line.split('+')[0]
        enh_path = line.split('+')[-1].strip()
        clean, fs1 = sf.read(speech_path)
        enh, fs2 = sf.read(enh_path)
        length = min(len(clean), len(enh))
        clean = clean[:length]
        enh = enh[:length]
        d = stoi(clean, enh, fs1, extended=False)
        save = ("%.4f\t" % d + enh_path + "\n")
        save_queque.put(save)
def calculate_ssnr(args, process_queue, save_queque):
    while True:
        if process_queue.qsize() % 2000 == 0:
            print("processed 2000 utterances, remain %d utterances" % process_queue.qsize())
            if process_queue.qsize() == 0:
                break
        line = process_queue.get()
        if line == None:
            break
        speech_path = line.split('+')[0]
        enh_path = line.split('+')[-1].strip()
        clean, fs1 = librosa.load(speech_path, sr=None)
        # print(fs)
        enh, fs2 = librosa.load(enh_path, sr=None)
        assert fs1 == fs2
        assert fs1 == 16000
        length = min(len(clean), len(enh))
        clean = clean[:length]
        enh = enh[:length]
        srate = 16000
        eps = 1e-10
        ref_wav = clean
        deg_wav = enh
        clean_speech = ref_wav
        processed_speech = deg_wav
        clean_length = ref_wav.shape[0]
        processed_length = deg_wav.shape[0]
        # Scale both to have same dynamic range. Remove DC too.
        dif = ref_wav - deg_wav
        overall_snr = 10 * np.log10(np.sum(ref_wav ** 2) / (np.sum(dif ** 2) + 10e-20))

        # Global variables
        winlength = int(np.round(30 * srate / 1000))  # 30 msecs
        skiprate = winlength // 4
        MIN_SNR = -10
        MAX_SNR = 35

        # For each frame, calculate SSNR
        num_frames = int(clean_length / skiprate - (winlength / skiprate))
        start = 0
        time = np.linspace(1, winlength, winlength) / (winlength + 1)
        window = 0.5 * (1 - np.cos(2 * np.pi * time))
        segmental_snr = []

        for frame_count in range(int(num_frames)):
            # (1) Get the frames for the test and ref speech
            # Apply Hanning Window
            clean_frame = clean_speech[start:start + winlength]
            processed_frame = processed_speech[start:start + winlength]
            clean_frame = clean_frame * window
            processed_frame = processed_frame * window

            # (2) Compute Segmental SNR
            signal_energy = np.sum(clean_frame ** 2)
            noise_energy = np.sum((clean_frame - processed_frame) ** 2)
            segmental_snr.append(10 * np.log10(signal_energy / (noise_energy + eps) + eps))
            segmental_snr[-1] = max(segmental_snr[-1], MIN_SNR)
            segmental_snr[-1] = min(segmental_snr[-1], MAX_SNR)
            start += int(skiprate)
        d = np.mean(segmental_snr)
        save = ("%.4f\t" % d + enh_path + "\n")
        save_queque.put(save)
def calculate_lsd(args, process_queue, save_queque):
    while True:
        if process_queue.qsize() % 2000 == 0:
            print("processed 2000 utterances, remain %d utterances" % process_queue.qsize())
            if process_queue.qsize() == 0:
                break
        line = process_queue.get()
        if line == None:
            break
        speech_path = line.split('+')[0]
        enh_path = line.split('+')[-1].strip()
        clean, fs1 = librosa.load(speech_path, sr=None)
        enh, fs2 = librosa.load(enh_path, sr=None)
        assert fs1 == fs2
        assert fs1 == 16000
        length = min(len(clean), len(enh))
        clean = clean[:length]
        enh = enh[:length]
        true_spectrogram = librosa.core.stft(clean, n_fft=512, \
                                       hop_length=int(512*0.5), \
                                        win_length=512, window='hann', center=False)
        reco_spectrogram = librosa.core.stft(enh, n_fft=512, \
                                       hop_length=int(512*0.5), \
                                        win_length=512, window='hann', center=False)
        # compute LSD
        true_X = 20 * np.log10(np.abs(true_spectrogram)+1e-10)
        reco_X = 20 * np.log10(np.abs(reco_spectrogram)+1e-10)
        reco_X_diff_squared = (true_X - reco_X) ** 2
        d = np.mean(np.sqrt(np.mean(reco_X_diff_squared[112:, :], axis=0)))

        save = ("%.4f\t" % d + enh_path + "\n")
        save_queque.put(save)
def calculate_sisdr(args, process_queue, save_queque):
    def remove_dc(signal):
        """Normalized to zero mean"""
        mean = np.mean(signal)
        signal -= mean
        return signal

    def pow_np_norm(signal):
        """Compute 2 Norm"""
        return np.square(np.linalg.norm(signal, ord=2))

    def pow_norm(s1, s2):
        return np.sum(s1 * s2)

    while True:
        if process_queue.qsize() % 2000 == 0:
            print("processed 2000 utterances, remain %d utterances" % process_queue.qsize())
            if process_queue.qsize() == 0:
                break
        line = process_queue.get()
        if line == None:
            break
        speech_path = line.split('+')[0]
        enh_path = line.split('+')[-1].strip()
        clean, fs1 = librosa.load(speech_path, sr=None)
        enh, fs2 = librosa.load(enh_path, sr=None)
        assert fs1 == fs2
        assert fs1 == 16000
        length = min(len(clean), len(enh))
        clean = clean[:length]
        enh = enh[:length]
        original = clean
        estimated = enh
        target = pow_norm(estimated, original) * original / pow_np_norm(original)
        noise = estimated - target
        d = 10 * np.log10(pow_np_norm(target) / pow_np_norm(noise))
        save = ("%.4f\t" % d + enh_path + "\n")
        save_queque.put(save)
def calculate_cd(args, process_queue, save_queque):
    def extract_overlapped_windows(x, nperseg, noverlap, window=None):
        # source: https://github.com/scipy/scipy/blob/v1.2.1/scipy/signal/spectral.py
        step = nperseg - noverlap
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
        strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                                 strides=strides)
        if window is not None:
            result = window * result
        return result

    def lpcoeff(speech_frame, model_order):
        eps = np.finfo(np.float64).eps
        # ----------------------------------------------------------
        # (1) Compute Autocorrelation Lags
        # ----------------------------------------------------------
        winlength = max(speech_frame.shape)
        R = np.zeros((model_order + 1,))
        for k in range(model_order + 1):
            if k == 0:
                R[k] = np.sum(speech_frame[0:] * speech_frame[0:])
            else:
                R[k] = np.sum(speech_frame[0:-k] * speech_frame[k:])

        # R=scipy.signal.correlate(speech_frame,speech_frame)
        # R=R[len(speech_frame)-1:len(speech_frame)+model_order]
        # ----------------------------------------------------------
        # (2) Levinson-Durbin
        # ----------------------------------------------------------
        a = np.ones((model_order,))
        a_past = np.ones((model_order,))
        rcoeff = np.zeros((model_order,))
        E = np.zeros((model_order + 1,))

        E[0] = R[0]

        for i in range(0, model_order):
            a_past[0:i] = a[0:i]

            sum_term = np.sum(a_past[0:i] * R[i:0:-1])

            if E[i] == 0.0:  # prevents zero division error, numba doesn't allow try/except statements
                rcoeff[i] = np.inf
            else:
                rcoeff[i] = (R[i + 1] - sum_term) / (E[i])

            a[i] = rcoeff[i]
            # if i==0:
            #    a[0:i] = a_past[0:i] - rcoeff[i]*np.array([])
            # else:
            if i > 0:
                a[0:i] = a_past[0:i] - rcoeff[i] * a_past[i - 1::-1]

            E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]

        acorr = R;
        refcoeff = rcoeff;
        lpparams = np.ones((model_order + 1,))
        lpparams[1:] = -a
        return (lpparams, R)

    def lpc2cep(a):
        #
        # converts prediction to cepstrum coefficients
        #
        # Author: Philipos C. Loizou

        M = len(a);
        cep = np.zeros((M - 1,));

        cep[0] = -a[1]

        for k in range(2, M):
            ix = np.arange(1, k)
            vec1 = cep[ix - 1] * a[k - 1:0:-1] * (ix)
            cep[k - 1] = -(a[k] + np.sum(vec1) / k);
        return cep

    while True:
        if process_queue.qsize() % 2000 == 0:
            print("processed 2000 utterances, remain %d utterances" % process_queue.qsize())
            if process_queue.qsize() == 0:
                break
        line = process_queue.get()
        if line == None:
            break
        speech_path = line.split('+')[0]
        enh_path = line.split('+')[-1].strip()
        clean, fs1 = librosa.load(speech_path, sr=None)
        enh, fs2 = librosa.load(enh_path, sr=None)
        assert fs1 == fs2
        assert fs1 == 16000
        length = min(len(clean), len(enh))

        fs = fs1
        frameLen = 0.03
        overlap = 0.75
        clean_speech = clean[:length]
        processed_speech = enh[:length]

        clean_length = len(clean_speech)
        processed_length = len(processed_speech)

        winlength = round(frameLen * fs)  # window length in samples
        skiprate = int(np.floor((1 - overlap) * frameLen * fs))  # window skip in samples

        if fs < 10000:
            P = 10  # LPC Analysis Order
        else:
            P = 16;  # this could vary depending on sampling frequency.

        C = 10 * np.sqrt(2) / np.log(10)

        numFrames = int(clean_length / skiprate - (winlength / skiprate));  # number of frames

        hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
        clean_speech_framed = extract_overlapped_windows(
            clean_speech[0:int(numFrames) * skiprate + int(winlength - skiprate)], winlength, winlength - skiprate,
            hannWin)
        processed_speech_framed = extract_overlapped_windows(
            processed_speech[0:int(numFrames) * skiprate + int(winlength - skiprate)], winlength, winlength - skiprate,
            hannWin)
        distortion = np.zeros((numFrames,))

        for ii in range(numFrames):
            A_clean, R_clean = lpcoeff(clean_speech_framed[ii, :], P)
            A_proc, R_proc = lpcoeff(processed_speech_framed[ii, :], P)

            C_clean = lpc2cep(A_clean)
            C_processed = lpc2cep(A_proc)
            distortion[ii] = min((10, C * np.linalg.norm(C_clean - C_processed)))

        IS_dist = distortion
        alpha = 0.95
        IS_len = round(len(IS_dist) * alpha)
        IS = np.sort(IS_dist)
        d = np.mean(IS[0: IS_len])


        save = ("%.4f\t" % d + enh_path + "\n")
        save_queque.put(save)

def generate_txtfile(args):
    directory = args. wav_directory
    txtfile_dir = args.txtfile_dir
    create_folder(os.path.dirname(txtfile_dir))
    if os.path.exists(txtfile_dir):
        os.remove(txtfile_dir)

    file_write_obj = open(txtfile_dir, 'w')
    for root, dirnames, filenames in os.walk(directory):
        if os.name == 'posix':  # check operation system: posix == liunx   nt == windows
            if root.split('/')[-1] == 'noise' or root.split('/')[-1] == 'clean':
                continue
        else:
            if root.split('\\')[-1] == 'noise' or root.split('\\')[-1] == 'clean':
                continue
        print("entering {} and dectecting {} utterances".format(root, len(filenames)))

        for file in filenames:
            gt_abs_path = os.path.join(os.path.join(directory, 'clean'), file)
            enh_abs_path = os.path.join(root, file)
            var = gt_abs_path + '+' + enh_abs_path
            file_write_obj.writelines(var)
            file_write_obj.write('\n')
    file_write_obj.close()
def queue_fill(args, manager):
    que = manager.Queue()
    txtfile_path = args.txtfile_dir

    f = open(txtfile_path, "r")
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        line = lines[i]
        speech_path = line.split('+')[0]
        enh_path = line.split('+')[-1].strip()
        if speech_path == enh_path:
            continue
        que.put(line)
    return que
def process_control_pesq(args):
    if os.path.exists(args.save_pesq_dir):
        os.remove(args.save_pesq_dir)

    cpus = mp.cpu_count()
    cpus = args.cpus
    pool = mp.Pool(processes=cpus)
    m = mp.Manager()
    process_queque = queue_fill(args, m)
    save_queque = m.Queue()
    for _ in range(32):
        process_queque.put(None)
    for i in range(0, cpus):
        pool.apply_async(calculate_pesq, args=(args, process_queque, save_queque))
    pool.close()
    pool.join()
    for _ in range(32):
        save_queque.put(None)
    output = open(args.save_pesq_dir, "w")
    while(1):
        line = save_queque.get()
        if line == None:
            break
        output.write(line)
    output.close()
    print("pesq proceess finish!")

def process_control_stoi(args):
    if os.path.exists(args.save_stoi_dir):
        os.remove(args.save_stoi_dir)
    cpus = mp.cpu_count()
    cpus = args.cpus
    pool = mp.Pool(processes=cpus)
    m = mp.Manager()
    process_queque = queue_fill(args, m)
    save_queque = m.Queue()

    for _ in range(32):
        process_queque.put(None)
    for i in range(0, cpus):
        pool.apply_async(calculate_stoi, args=(args, process_queque, save_queque))
    pool.close()
    pool.join()
    for _ in range(32):
        save_queque.put(None)
    output = open(args.save_stoi_dir, "w")
    while(1):
        line = save_queque.get()
        if line == None:
            break
        output.write(line)
    output.close()
    print("stoi proceess finish!")

def process_control_ssnr(args):
    if os.path.exists(args.save_ssnr_dir):
        os.remove(args.save_ssnr_dir)

    cpus = mp.cpu_count()
    cpus = args.cpus
    pool = mp.Pool(processes=cpus)
    m = mp.Manager()
    process_queque = queue_fill(args, m)
    save_queque = m.Queue()
    for _ in range(32):
        process_queque.put(None)
    for i in range(0, cpus):
        pool.apply_async(calculate_ssnr, args=(args, process_queque, save_queque))
    pool.close()
    pool.join()
    for _ in range(32):
        save_queque.put(None)
    output = open(args.save_ssnr_dir, "w")
    while(1):
        line = save_queque.get()
        if line == None:
            break
        output.write(line)
    output.close()
    print("ssnr proceess finish!")

def process_control_lsd(args):
    if os.path.exists(args.save_lsd_dir):
        os.remove(args.save_lsd_dir)

    cpus = mp.cpu_count()
    cpus = args.cpus
    pool = mp.Pool(processes=cpus)
    m = mp.Manager()
    process_queque = queue_fill(args, m)
    save_queque = m.Queue()
    for _ in range(32):
        process_queque.put(None)
    for i in range(0, cpus):
        pool.apply_async(calculate_lsd, args=(args, process_queque, save_queque))
    pool.close()
    pool.join()
    for _ in range(32):
        save_queque.put(None)
    output = open(args.save_lsd_dir, "w")
    while(1):
        line = save_queque.get()
        if line == None:
            break
        output.write(line)
    output.close()
    print("lsd proceess finish!")

def process_control_sisdr(args):
    if os.path.exists(args.save_sisdr_dir):
        os.remove(args.save_sisdr_dir)
    cpus = mp.cpu_count()
    cpus = args.cpus
    pool = mp.Pool(processes=cpus)
    m = mp.Manager()
    process_queque = queue_fill(args, m)
    save_queque = m.Queue()
    for _ in range(32):
        process_queque.put(None)
    for i in range(0, cpus):
        pool.apply_async(calculate_sisdr, args=(args, process_queque, save_queque))
    pool.close()
    pool.join()
    for _ in range(32):
        save_queque.put(None)
    output = open(args.save_sisdr_dir, "w")
    while(1):
        line = save_queque.get()
        if line == None:
            break
        output.write(line)
    output.close()
    print("sisdr proceess finish!")

def process_control_cd(args):
    if os.path.exists(args.save_cd_dir):
        os.remove(args.save_cd_dir)

    cpus = mp.cpu_count()
    cpus = args.cpus
    pool = mp.Pool(processes=cpus)
    m = mp.Manager()
    process_queque = queue_fill(args, m)
    save_queque = m.Queue()
    for _ in range(32):
        process_queque.put(None)
    for i in range(0, cpus):
        pool.apply_async(calculate_cd, args=(args, process_queque, save_queque))
    pool.close()
    pool.join()

    for _ in range(32):
        save_queque.put(None)
    output = open(args.save_cd_dir, "w")
    while(1):
        line = save_queque.get()
        if line == None:
            break
        output.write(line)
    output.close()
    print("cd proceess finish!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_directory', type=str, default='None')
    parser.add_argument('--txtfile_dir', type=str, default='None')
    parser.add_argument('--save_pesq_dir', type=str, default='None')
    args = parser.parse_args()
    path = os.getcwd()

    args.cpus = 6
    #args.wav_directory = os.path.join(path, 'enhan')
    args.wav_directory = os.path.join(path,'Output_data/test')
    args.txtfile_dir = os.path.join(path, 'result', 'cleanenh.txt')


    args.save_pesq_dir = os.path.join(path,'result', '_pesq_results.txt')
    args.save_stoi_dir = os.path.join(path,'result','_stoi_results.txt')
    args.save_ssnr_dir = os.path.join(path,'result','_ssnr_results.txt')
    args.save_lsd_dir = os.path.join(path,'result','_lsd_results.txt')
    args.save_sisdr_dir = os.path.join(path,'result','_sisdr_results.txt')
    args.save_cd_dir = os.path.join(path,'result','_cd_results.txt')


    # if os.path.exists(os.path.join(path,'result')):
    #     shutil.rmtree(os.path.join(path,'result'))
    if not os.path.exists(os.path.join(path,'result')):
        os.makedirs(os.path.join(path,'result'))
    print(json.dumps(vars(args), indent=2))
    generate_txtfile(args)
    process_control_pesq(args)
    process_control_stoi(args)
    # process_control_ssnr(args)
    # process_control_lsd(args)
    process_control_sisdr(args)
    # process_control_cd(args)


    print("\n***************ob_test_mult finish!******************\n")
