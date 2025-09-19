from scipy import interpolate
from scipy.signal import get_window
import math
import sys
import os
import numpy as np

HOP_SIZE = 128
SAMPLING_RATE = 44100
WINDOW_SIZE = 1025  # int(2*(((1024/16000)*SAMPLING_RATE)//2))-1
WINDOW_TYPE = 'blackmanharris'

sms_tools_path = 'sms_tools'  # if you use git clone to this dir (e.g. in colab)
# change according to the install location of https://github.com/MTG/sms-tools

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), sms_tools_path, './models/'))
try:
    import sineModel as SM
    import harmonicModel as HM
except ImportError:
    print ("Please refer to the README.md file in the 'sms-tools' directory,")
    print ("Error importing sms-tools. Check if the sms path is correct.")
    print ("\n")
    sys.exit(0)


def interpolate_f0_to_sr(pitch_track_csv, audio, sr=SAMPLING_RATE, hop_size=HOP_SIZE):
    f = interpolate.interp1d(pitch_track_csv["time"],
                             pitch_track_csv["frequency"],
                             kind="nearest", fill_value="extrapolate")
    c = interpolate.interp1d(pitch_track_csv["time"],
                             pitch_track_csv["confidence"],
                             kind="nearest", fill_value="extrapolate")
    start_frame = 0  # I was true at first! It starts from zero!! int(np.floor((WINDOW_SIZE + 1) / 2))
    end_frame = len(audio) - (len(audio) % hop_size) + start_frame
    time = np.array(range(start_frame, end_frame + 1, hop_size)) / sr
    pitch_track_np = f(time)
    confidence_np = c(time)
    pitch_track_np[pitch_track_np < 10] = 0  # interpolation might introduce odd frequencies
    return pitch_track_np, confidence_np, time


def anal(audio, f0, n_harmonics=30, hop_size=HOP_SIZE, sr=SAMPLING_RATE):
    # 验证 f0 数组不为空且包含有效数据
    if f0 is None or len(f0) == 0 or np.all(f0 <= 0):
        print("Invalid or empty F0 data.")
        return np.array([]), np.array([]), np.array([])
    # Get harmonic content from audio using extracted pitch as reference
    w = get_window(WINDOW_TYPE, WINDOW_SIZE, fftbins=True)
    hfreq, hmag, hphase = harmonicModelAnal(
        x=audio,
        f0=f0,
        fs=sr,
        w=w,
        H=hop_size,
        N=2048,
        t=-90,
        nH=n_harmonics,
        #harmDevSlope=0.001,
        harmDevSlope=0.01,
        minSineDur=0.001
    )
    return hfreq, hmag, hphase


def harmonicModelAnal(x, f0, fs, w, N, H, t, nH, harmDevSlope=0.01, minSineDur=.02):
    """
	  Analysis of a sound using the sinusoidal harmonic model based on given f0s.
    x: input sound; fs: sampling rate, w: analysis window; N: FFT size (minimum 512); t: threshold in negative dB,
    nH: maximum number of harmonics;  minf0: minimum f0 frequency in Hz,
    maxf0: maximim f0 frequency in Hz; f0et: error threshold in the f0 detection (ex: 5),
    harmDevSlope: slope of harmonic deviation; minSineDur: minimum length of harmonics
    returns xhfreq, xhmag, xhphase: harmonic frequencies, magnitudes and phases
    """

    if (minSineDur < 0):  # raise exception if minSineDur is smaller than 0
        raise ValueError("Minimum duration of sine tracks smaller than 0")

    hN = N // 2  # size of positive spectrum
    hM1 = int(math.floor((w.size + 1) / 2))  # half analysis window size by rounding
    hM2 = int(math.floor(w.size / 2))  # half analysis window size by floor
    x = np.append(np.zeros(hM2), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hM2))  # add zeros at the end to analyze last sample
    pin = hM1  # init sound pointer in middle of anal window
    pend = x.size - hM1  # last sample to start a frame
    w = w / sum(w)  # normalize analysis window
    hfreqp = []  # initialize harmonic frequencies of previous frame
    f0_idx = 0
    while pin <= pend:
        x1 = x[pin - hM1:pin + hM2]  # select frame
        mX, pX = HM.DFT.dftAnal(x1, w, N)  # compute dft
        ploc = HM.UF.peakDetection(mX, t)  # detect peak locations
        iploc, ipmag, ipphase = HM.UF.peakInterp(mX, pX, ploc)  # refine peak values
        ipfreq = fs * iploc / N  # convert locations to Hz
        # Example check in harmonicModelAnal before calling harmonicDetection
        # if len(ipfreq) == 0 or f0[f0_idx] <= 0:
        #     # Skip this frame or handle appropriately
        #     continue
        hfreq, hmag, hphase = HM.harmonicDetection(ipfreq, ipmag, ipphase,
                                                   f0[f0_idx], nH, hfreqp, fs,
                                                   harmDevSlope)  # find harmonics
        hfreqp = hfreq
        if pin == hM1:  # first frame
            xhfreq = np.array([hfreq])
            xhmag = np.array([hmag])
            xhphase = np.array([hphase])
        else:  # next frames
            xhfreq = np.vstack((xhfreq, np.array([hfreq])))
            xhmag = np.vstack((xhmag, np.array([hmag])))
            xhphase = np.vstack((xhphase, np.array([hphase])))
        pin += H
        f0_idx += 1
    xhfreq = SM.cleaningSineTracks(xhfreq, round(fs * minSineDur / H))  # delete tracks shorter than minSineDur
    return xhfreq, xhmag, xhphase


def synth(hfreqs, hmags, hphases, N=512, H=HOP_SIZE, fs=SAMPLING_RATE):
    return SM.sineModelSynth(hfreqs, hmags, hphases, N=N, H=H, fs=fs)

