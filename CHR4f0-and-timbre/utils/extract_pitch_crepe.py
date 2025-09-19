import os
import glob
import numpy as np
import librosa
import crepe
from scipy.ndimage import gaussian_filter1d
from librosa.sequence import viterbi_discriminative
import pandas as pd


def predict(audio, sr, model_capacity='full',
            viterbi=False, center=True, step_size=10, verbose=1):
    """
    Perform pitch estimation on given audio

    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N, C)]
        The audio samples. Multichannel audio will be downmixed.
    sr : int
        Sample rate of the audio samples. The audio will be resampled if
        the sample rate is not 16 kHz, which is expected by the model.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    viterbi : bool
        Apply viterbi smoothing to the estimated pitch curve. False by default.
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : int
        Set the keras verbosity mode: 1 (default) will print out a progress bar
        during prediction, 0 will suppress all non-error printouts.

    Returns
    -------
    A 4-tuple consisting of:

        time: np.ndarray [shape=(T,)]
            The timestamps on which the pitch was estimated
        frequency: np.ndarray [shape=(T,)]
            The predicted pitch values in Hz
        confidence: np.ndarray [shape=(T,)]
            The confidence of voice activity, between 0 and 1
        activation: np.ndarray [shape=(T, 360)]
            The raw activation matrix
    """
    activation = crepe.core.get_activation(audio, sr, model_capacity=model_capacity,
                                           center=center, step_size=step_size,
                                           verbose=verbose)

    if viterbi:
        # NEW!! CONFIDENCE IS NO MORE THE MAX ACTIVATION! CORRECTED TO BE CALCULATED ALONG THE PATH!
        path, cents = to_viterbi_cents(activation)
        confidence = np.array([activation[i, path[i]] for i in range(len(activation))])
    else:
        cents = crepe.core.to_local_average_cents(activation)
        confidence = activation.max(axis=1)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0

    time = np.arange(confidence.shape[0]) * step_size / 1000.0

    return time, frequency, confidence, activation


def to_viterbi_cents(salience):
    """
    Find the Viterbi path using a transition prior that induces pitch
    continuity.
    """
    # transition probabilities inducing continuous pitch
    # big changes are penalized with one order of magnitude
    transition = gaussian_filter1d(np.eye(360), 30) + 9 * gaussian_filter1d(np.eye(360), 2)
    transition = transition / np.sum(transition, axis=1)[:, None]

    p = salience / salience.sum(axis=1)[:, None]
    p[np.isnan(p.sum(axis=1)), :] = np.ones(360) * 1 / 360
    path = viterbi_discriminative(p.T, transition)

    return path, np.array([crepe.core.to_local_average_cents(salience[i, :], path[i]) for i in
                           range(len(path))])


def predict_from_file_list(audio_files, output_f0_files, viterbi=True):
    for index, audio_file in enumerate(audio_files):
        output_f0_file = output_f0_files[index]
        audio, sr = librosa.load(audio_file, mono=True)
        time, frequency, confidence, activation = predict(audio, sr, viterbi=viterbi)
        df = pd.DataFrame({"time": time, "frequency": frequency, "confidence": confidence},
                          columns=["time", "frequency", "confidence"])
        df.to_csv(output_f0_file, index=False)
    return


def extract_pitch_crepe(main_dataset_folder, viterbi=True):

    out_folder = os.path.join(main_dataset_folder, 'pitch_tracks', "crepe")
    audio_files = sorted(glob.glob(os.path.join(main_dataset_folder, "original", "**")))
    output_f0_files = []
    for file in audio_files:
        out_f0 = os.path.relpath(file[:-3] + "f0.csv", os.path.join(main_dataset_folder, "original"))
        output_f0_files.append(os.path.join(out_folder, out_f0))
    for file in output_f0_files:
        parent = os.path.dirname(file)
        if not os.path.exists(parent):
            # Create a new directory because it does not exist
            os.makedirs(parent)
    predict_from_file_list(audio_files, output_f0_files, viterbi=viterbi)
    return


if __name__ == '__main__':
    #dataset_folder = "/run/user/1000/gvfs/sftp:host=hpc.s.upf.edu/homedtic/ntamer/musical-etudes/clarinet-etudes"
    #dataset_folder = "/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/flute-etudes"
    dataset_folder = "/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo"
    extract_pitch_crepe(dataset_folder, viterbi=True)
