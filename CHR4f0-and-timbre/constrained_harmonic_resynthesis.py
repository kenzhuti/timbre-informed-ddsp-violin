import os
import pickle

import resynthesis_utils
import silencing_constraints
import numpy as np
import librosa
import pandas as pd
import joblib
from time import time as taymit
import soundfile as sf
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import random

import traceback


HOP_SIZE = 128
SAMPLING_RATE = 44100
WINDOW_SIZE = 1025  # int(2*(((1024/16000)*SAMPLING_RATE)//2))-1
WINDOW_TYPE = 'blackmanharris'
LOWEST_NOTE_ALLOWED_HZ = 180


def analyze_file(paths, confidence_threshold=0.6, min_voiced_segment_ms=25):
    if os.path.exists(paths['anal']):
        print(f"File {paths['anal']} already exists. Skipping analysis.")
        return
    print("start analyzing...")
    time_start = taymit()
    filename = ' '.join(paths['original'].split('/')[-3:])[:-4]
    audio = librosa.load(paths['original'], sr=SAMPLING_RATE, mono=True)[0]
    f0s = pd.read_csv(paths['f0'])
    if f0s.columns.values[0] != 'time':
        f0s = pd.read_csv(paths['f0'], header=None, names=['time', 'f0'])
    f0s, conf, time = resynthesis_utils.interpolate_f0_to_sr(f0s, audio)
    time_load = taymit()
    print("loading {:s} took {:.3f}".format(filename, time_load - time_start))
    hfreq, hmag, _ = resynthesis_utils.anal(audio, f0s, n_harmonics=25) # 12->25 for violin
    f0s = f0s[:len(hmag)]
    conf = conf[:len(hmag)]
    time = time[:len(hmag)]
    time_anal = taymit()

    conf_bool = conf > confidence_threshold
    conf_bool_1 = conf < 1.0
    conf_bool = np.logical_and(conf_bool, conf_bool_1)
    valid_f0_bool = f0s > LOWEST_NOTE_ALLOWED_HZ
    # lowest note on violin is G3 = 196 hz, so threshold with sth close to the lowest note
    valid_hmag_bool = (hmag > -100).sum(axis=1) > 3  # at least three harmonics
    valid_bool = np.logical_and(valid_hmag_bool, valid_f0_bool)
    valid_bool = np.logical_and(conf_bool, valid_bool)
    min_voiced_segment_len = int(np.ceil((min_voiced_segment_ms / 1000) / (HOP_SIZE / SAMPLING_RATE)))
    valid_bool = silencing_constraints.silence_segments_one_run(valid_bool, 0, min_voiced_segment_len)  # if keeps high for some duration

    print("anal {:s} took {:.3f}. coverage: {:.3f}".format(filename, time_anal - time_load,
                                                           sum(valid_bool) / len(valid_bool)))
    np.savez_compressed(paths['anal'], time=time[valid_bool], f0=f0s[valid_bool], hfreq=hfreq[valid_bool, :25], hmag=hmag[valid_bool, :25]) # 12->25 for violin
    return


def synth_file(paths, instrument_detector=None, refine_twm=True, pitch_shift=False,
               th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, create_tfrecords=False, synth="resynth",
               parse="norm", hmag_std_scl=None, inharm=None,
               scaler_path=None, kmeans_model_path=None):
    # Check if both synthesized files already exist
    if os.path.exists(paths['synth']) and os.path.exists(paths['synth_f0']):
        print(f"Files {paths['synth']} and {paths['synth_f0']} already exist. Skipping synthesis.")
        return

    time_start = taymit()
    print("start...")

    audio = librosa.load(paths['original'], sr=SAMPLING_RATE, mono=True)[0]
    f0s = pd.read_csv(paths['f0'])
    filename = ' '.join(paths['original'].split('/')[-3:])[:-4]
    try:
        f0s["confidence"] = f0s["confidence"].fillna(0)
        pre_anal_coverage = f0s['confidence'] > th_lc
        pre_anal_coverage = sum(pre_anal_coverage) / len(pre_anal_coverage)
        print("pre_anal_coverage: ", pre_anal_coverage)
        # f0s = silencing_constraints.silence_unvoiced_segments(f0s, low_confidence_threshold=th_lc,
        #                                                       high_confidence_threshold=th_hc,
        #                                                       min_voiced_segment_ms=voiced_th_ms)
        # f0s = apply_pitch_filter(f0s, min_chunk_size=21, median=True, confidence_threshold=th_hc)
        f0s, conf, time = resynthesis_utils.interpolate_f0_to_sr(f0s, audio)
        time_load = taymit()
        print("loading {:s} took {:.3f}".format(filename, time_load - time_start))

        if synth == "tb":
            print("synthesizing with timbre clustering...")
            f0s, hfreqs, hmags, hphases = parse_timbre_cluster(paths['timbre_cluster'], paths['anal'],
                                                               parse=parse, scaler_path=scaler_path,
                                                               kmeans_model_path=kmeans_model_path)
        elif synth == "anal":
            print("synthesizing with anal npz file...")
            f0s, hfreqs, hmags, hphases = parse_anal_npz(paths['anal'])
        else: # synth == "resynth"
            print("synthesizing with original resynthesis protocol...")
            hfreqs, hmags, hphases = resynthesis_utils.anal(audio, f0s, n_harmonics=25) #40
        f0s = f0s[:len(hmags)]
        conf = conf[:len(hmags)]
        time = time[:len(hmags)]
        time_anal = taymit()
        print("anal {:s} took {:.3f}".format(filename, time_anal - time_load))
        # if instrument_detector is not None:
        #     hfreqs, hmags, hphases, f0 = silencing_constraints.supress_timbre_anomalies(instrument_detector,
        #                                                                                 hfreqs, hmags, hphases, f0s)
        # if refine_twm:
        #     hfreqs, hmags, hphases, f0s = silencing_constraints.refine_harmonics_twm(hfreqs, hmags, hphases,
        #                                                                              f0s, f0et=5.0,
        #                                                                              f0_refinement_range_cents=16,
        #                                                                              min_voiced_segment_ms=voiced_th_ms)
        time_refine = taymit()
        post_anal_coverage = sum(f0s > 0) / len(f0s)
        print("post_anal_coverage: ", post_anal_coverage)
        coverage = post_anal_coverage / pre_anal_coverage
        print("refining parameters for {:s} took {:.3f}. coverage: {:.3f}".format(filename,
                                                                                  time_refine - time_anal,
                                                                                  coverage))
        # if sawtooth_synth:
        #     hmags[f0s > 0] = -30 - 20 * np.log10(np.arange(1, 41))
        #     hfreqs[f0s > 0] = np.dot(hfreqs[f0s > 0][:, 0][:, np.newaxis], np.arange(1, 41)[np.newaxis, :])
        #     hphases = np.array([])
        harmonic_audio = resynthesis_utils.synth(hfreqs, hmags, hphases, N=512, H=HOP_SIZE, fs=SAMPLING_RATE)
        # Ensure the directory for 'synth' exists
        synth_dir = os.path.dirname(paths['synth'])
        if not os.path.exists(synth_dir):
            os.makedirs(synth_dir)

        # Ensure the directory for 'synth_f0' exists
        synth_f0_dir = os.path.dirname(paths['synth_f0'])
        if not os.path.exists(synth_f0_dir):
            os.makedirs(synth_f0_dir)

        sf.write(paths['synth'], harmonic_audio, 44100, 'PCM_24')
        df = pd.DataFrame([time, f0s]).T
        df.to_csv(paths['synth_f0'], header=False, index=False,
                  float_format='%.6f')
        if create_tfrecords:
            tfrecord_file(paths, 'synth')
        if pitch_shift:
            sign = random.choice([-1, 1])
            val = random.choice(range(5, 50))
            pitch_shift_cents = sign * val

            alt_f0s = f0s * pow(2, (pitch_shift_cents / 1200))
            # Synthesize audio with the shifted harmonic content
            alt_hfreqs = hfreqs * pow(2, (pitch_shift_cents / 1200))
            alt_harmonic_audio = resynthesis_utils.synth(alt_hfreqs, hmags, np.array([]), N=512,
                                                         H=HOP_SIZE, fs=SAMPLING_RATE)
            sf.write(paths['shifted'], alt_harmonic_audio,
                     44100, 'PCM_24')
            df = pd.DataFrame([time, alt_f0s]).T
            df.to_csv(paths['shifted_f0'], header=False, index=False, float_format='%.6f')
            if create_tfrecords:
                tfrecord_file(paths, 'shifted')

        time_synth = taymit()
        print("synthesizing {:s} took {:.3f}. Total resynthesis took {:.3f}".format(filename, time_synth - time_refine,
                                                                                    time_synth - time_load))
    except Exception as e:
        print(f"Error processing {paths['original']}: {e}")
        traceback.print_exc()  # Print detailed stack trace information
    return


def tfrecord_file(paths, name='synth'):
    #import tensorflow.compat.v1 as tf
    import tensorflow as tf
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    labels = np.loadtxt(paths[name+'_f0'], delimiter=',')

    nonzero = labels[:, 1] > 0
    absdiff = np.abs(np.diff(np.concatenate(([False], nonzero, [False]))))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    nonzero[ranges[:, 0]] = False
    nonzero[ranges[:, 1]-1] = False
    # get rid of the boundary points since it may contain some artifacts
    # since the hop size in synthesis is 2.9 ms it roughly corresponds to 512/16000
    labels = labels[nonzero, :]

    if len(labels):
        sr = 16000
        audio = librosa.load(paths[name], sr=sr)[0]

        output_path = paths[name+'_tfrecord']
        writer = tf.python_io.TFRecordWriter(output_path, options=options)

        for row in tqdm(labels):
            pitch = row[1]
            center = int(row[0] * sr)
            segment = audio[center - 512:center + 512]
            if len(segment) == 1024:
                example = tf.train.Example(features=tf.train.Features(feature={
                    "audio": tf.train.Feature(float_list=tf.train.FloatList(value=segment)),
                    "pitch": tf.train.Feature(float_list=tf.train.FloatList(value=[pitch]))
                }))
                writer.write(example.SerializeToString())
        writer.close()
    return

def inverse_preprocess_hmag(
    X_proc: np.ndarray,
    scaler: MinMaxScaler or QuantileTransformer or None,
    log: bool = False,
    min_val: float = -100
) -> np.ndarray:
    """
    Inverse preprocessing function to revert hmag features back to original dBFS scale.

    Parameters
    ----------
    X_proc : np.ndarray
        Preprocessed hmag array (frames x harmonics).
    scaler : MinMaxScaler or None
        The fitted scaler object used during preprocessing.
    log : bool
        If True, assumes log1p transformation was applied and reverses it.
    min_val : float
        The minimum value subtracted during log transformation, default -100.

    Returns
    -------
    X_orig : np.ndarray
        Reverted hmag array in dBFS scale (frames x harmonics).
    """
    if scaler is not None:
        X = scaler.inverse_transform(X_proc)
    else:
        X = X_proc.copy()

    if log:
        X = np.expm1(X) + min_val

    return X


def inverse_standardize_hmags(standardized_hmags, means, scales):
    return standardized_hmags * scales + means


def inverse_minmax_scale_hmags(scaled_hmags, data_min, data_max):
    return scaled_hmags * (data_max - data_min) + data_min


def inverse_normalize_hmags(normalized_hmags, weights):
    return normalized_hmags * weights[:, np.newaxis]

def inverse_transform(hmag_std_transformed, K, mean, scale):
    # Inverse StandardScaler
    hmag_reflect_sqrt_transformed = hmag_std_transformed * scale + mean

    # Inverse Reflect and Square Root Transformation
    hmag_origin = K - (hmag_reflect_sqrt_transformed ** 2)

    return hmag_origin


def decode_with_gaussian_variability(mapped_features, sigma, add_gaussian_variability=True, mu=0):
    if add_gaussian_variability:
        # Perturb the centroids with Gaussian noise
        perturbed_features = mapped_features + np.random.normal(mu, sigma, mapped_features.shape)
    else:
        # Use the centroids without any added noise
        perturbed_features = mapped_features
    return perturbed_features


def parse_timbre_cluster(timbre_cluster_file, anal_file, parse="norm", scaler_path=None, kmeans_model_path=None):
    SIGMA = 0.24961838438770378 # Calculated from ../sample_data
    base_dir = os.path.dirname(timbre_cluster_file)
    data = pd.read_csv(timbre_cluster_file)

    f0s = data['f0'].values # f0s = data['f0']
    if 'labels' in data.columns:
        indices = data['labels'].values.astype(int)
    elif 'label' in data.columns:
        indices = data['label'].values.astype(int)
    else:
        raise ValueError("The data file must contain either 'labels' or 'indices' column.")

    if parse == "std":
        centroid_file = os.path.join(base_dir, 'centroids_std.csv')
        stats_file = os.path.join(base_dir, 'standardization_stats.csv')
        stats = pd.read_csv(stats_file)
        means = stats['mean'].values
        scales = stats['scale'].values
        weights = None

    elif parse == "scl":
        centroid_file = os.path.join(base_dir, 'centroids_minmax.csv')
        stats_file = os.path.join(base_dir, 'minmax_stats.csv')
        stats = pd.read_csv(stats_file)
        data_min = stats['data_min'].values
        data_max = stats['data_max'].values
        weights = None

    elif parse == "norm":
        centroid_file = os.path.join(base_dir, 'centroids_normalized.csv')
        weights = data['weights'].values

    elif parse == "std-trans":
        centroid_file = os.path.join(base_dir, 'centroids_transform_std.csv')
        stats_file = os.path.join(base_dir, 'transform_standardization_stats.csv')
        stats = pd.read_csv(stats_file)
        means = stats['mean'].values
        scales = stats['scale'].values
        K = stats['K_value'].values[0]

    elif parse == "minmax_log" or parse == "quantile":
        if scaler_path is None or kmeans_model_path is None:
            raise ValueError("scaler_path and kmeans_model_path must be provided for minmax_log or quantile parse.")
        kmeans = joblib.load(kmeans_model_path)
        scaler = joblib.load(scaler_path)
        centroid = kmeans.cluster_centers_
        weights = None

    else:
        raise ValueError("Invalid parse option. Choose from 'std', 'scl', 'norm', 'minmax_log' or 'quantile'.")

    if parse != "minmax_log" and parse != "quantile":
        centroid = pd.read_csv(centroid_file)
        n_harmonics = centroid.shape[1] - 1
        centroid_index = centroid['centroid_index']
        #features = centroid.loc[:, 'feature_0':'feature_24']
        features = centroid.iloc[:, :n_harmonics]

        # Create a dictionary mapping from centroid_index to features
        feature_dict = {index: row for index, row in zip(centroid_index, features.values)}

        # Map indices in data to corresponding features in centroid
        mapped_features = np.array([feature_dict[index] for index in indices])
    else:
        n_harmonics = centroid.shape[1]
        feature_dict = {i: centroid[i] for i in range(centroid.shape[0])}
        mapped_features = np.array([feature_dict[index] for index in indices])

    # Parse hmags based on the parsing type
    if parse == "std":
        hmags = inverse_standardize_hmags(mapped_features, means, scales)
    elif parse == "scl":
        hmags = inverse_minmax_scale_hmags(mapped_features, data_min, data_max)
    elif parse == "norm":
        hmags = inverse_normalize_hmags(mapped_features, weights)
    elif parse == "std-trans":
        hmags = inverse_transform(mapped_features, K, means, scales)
    elif parse == "minmax_log":
        hmags = inverse_preprocess_hmag(mapped_features, scaler, log=True, min_val=-100)
    elif parse == "quantile":
        hmags = inverse_preprocess_hmag(mapped_features, scaler, log=False)


    # Clamp hmags within [-100, -4.9325553959965465]
    hmags = np.clip(hmags, -100, -4.9325553959965465)

    # Calculate hfreq
    # Perfect harmonics
    #hfreq = np.array(f0s)[:, np.newaxis] * np.arange(1, n_harmonics + 1)

    # Real harmonics from *.anal files
    anal_data = np.load(anal_file)
    hfreq = anal_data['hfreq']

    # Create empty hphase, which can be handled in line 210 in sineModel.py for generation
    hphase = np.array([])

    return f0s, hfreq, hmags, hphase


def parse_anal_npz(anal_file):
    data = np.load(anal_file)
    f0s = data['f0']
    hmag = data['hmag']
    hfreq = data['hfreq']

    # Calculate PERFECT hfreq
    # n_harmonics = hmag.shape[1]
    # hfreq = np.array(f0s)[:, np.newaxis] * np.arange(1, n_harmonics + 1)

    hphase = np.array([])

    return f0s, hfreq, hmag, hphase


if __name__ == '__main__':
    # paths = {
    #     'original': '/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/original/1355vsSCpZ4tw00000217.mp3',
    #     'f0': '/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/pitch_tracks/crepe/1355vsSCpZ4tw00000217.f0.csv',
    #     'synth': '/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/resynth/ic_100_0.05_crepe/1355vsSCpZ4tw00000217.RESYN.wav',
    #     'synth_f0': '/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/resynth/ic_100_0.05_crepe/1355vsSCpZ4tw00000217.RESYN.csv',
    #     'shifted': '/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/resynth/ic_100_0.05_crepe/1355vsSCpZ4tw00000217.shiftedRESYN.wav',
    #     'shifted_f0': '/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/resynth/ic_100_0.05_crepe/1355vsSCpZ4tw00000217.shiftedRESYN.csv',
    #     'anal': '/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/anal/crepe/1355vsSCpZ4tw00000217.npz'
    # }
    #
    # instrument_model_file = "/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/ic_100_0.05.pkl"
    # with open(instrument_model_file, 'rb') as modelfile:
    #     instrument_timbre_detector = pickle.load(modelfile)
    # synth_file(paths, instrument_detector=instrument_timbre_detector, refine_twm=True, pitch_shift=False,
    #            th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, create_tfrecords=False)

    # First example
    f_basename = "1355vsSCpZ4tw00000217"
    #f_basename = "15f1ORtWqE6sQ00000124"
    #f_basename = "01AvUPt4ybvMM00000060"
    paths_tb = {
        'original': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/original/{f_basename}.mp3',
        'f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/pitch_tracks/crepe/{f_basename}.f0.csv',

        'synth': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/resynth/ic_100_0.05_crepe/{f_basename}.RESYN.wav',
        'synth_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/resynth/ic_100_0.05_crepe/{f_basename}.RESYN.csv',
        'shifted': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/resynth/ic_100_0.05_crepe/{f_basename}.shiftedRESYN.wav',
        'shifted_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/resynth/ic_100_0.05_crepe/{f_basename}.shiftedRESYN.csv',

        'anal': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/anal/crepe/{f_basename}.npz',
        'timbre_cluster': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/timber_cluster_88/{f_basename}.tb.csv'
    }

    # f_basename = "03byLdNJnbcss00000187"
    # f_basename = "03KkVk8TYzluo00000164"
    # f_basename = "03TAoWzc23xZg00000184"
    # f_basename = "03un_J2R2G-c400000094"
    # f_basename = "04fdEVMZicC6A00000111"
    # f_basename = "04HHoYqjiRsAs00000119"
    # f_basename = "12qkXtU7vscRQ00000061"
    # f_basename = "13j2q2OX1jo1s00000099"
    # f_basename = "15k9ls6YgC7WY00000153"
    # f_basename = "19Df_oVHUiD5800040118"

    paths_tb_std = {
        'original': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/original/{f_basename}.mp3',
        'f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/pitch_tracks/crepe/{f_basename}.f0.csv',

        'synth': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_std/resynth/{f_basename}.RESYN.wav',
        'synth_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_std/resynth/{f_basename}.RESYN.csv',
        'shifted': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_std/resynth/{f_basename}.shiftedRESYN.wav',
        'shifted_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_std/resynth/{f_basename}.shiftedRESYN.csv',

        'anal': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/anal/crepe/{f_basename}.npz',
        'timbre_cluster': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_std/{f_basename}.tb.csv'
    }

    paths_tb_scl = {
        'original': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/original/{f_basename}.mp3',
        'f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/pitch_tracks/crepe/{f_basename}.f0.csv',

        'synth': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/{f_basename}.RESYN.wav',
        'synth_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/{f_basename}.RESYN.csv',
        'shifted': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/{f_basename}.shiftedRESYN.wav',
        'shifted_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/{f_basename}.shiftedRESYN.csv',

        'anal': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/anal/crepe/{f_basename}.npz',
        'timbre_cluster': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/{f_basename}.tb.csv'
    }

    paths_tb_norm = {
        'original': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/original/{f_basename}.mp3',
        'f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/pitch_tracks/crepe/{f_basename}.f0.csv',

        'synth': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_normalized/resynth/{f_basename}.RESYN.wav',
        'synth_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_normalized/resynth/{f_basename}.RESYN.csv',
        'shifted': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_normalized/resynth/{f_basename}.shiftedRESYN.wav',
        'shifted_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_normalized/resynth/{f_basename}.shiftedRESYN.csv',

        'anal': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/anal/crepe/{f_basename}.npz',
        'timbre_cluster': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_normalized/{f_basename}.tb.csv'
    }

    #tb_cluster_file = f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/timber_cluster_88/{f_basename}.tb.csv'
    # print(parse_timbre_cluster(tb_cluster_file))


    instrument_model_file = "/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/ic_100_0.05.pkl"
    # with open(instrument_model_file, 'rb') as modelfile:
    #     instrument_timbre_detector = pickle.load(modelfile)
    # synth_file(paths_tb_std, instrument_detector=instrument_timbre_detector, refine_twm=True, pitch_shift=False,
    #            th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, create_tfrecords=False, synth="tb",
    #            parse="std") # synth: "tb", "anal", "resynth"; parse (only for synth="tb"): "std", "scl", "norm"
    #
    # synth_file(paths_tb_scl, instrument_detector=instrument_timbre_detector, refine_twm=True, pitch_shift=False,
    #            th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, create_tfrecords=False, synth="tb",
    #            parse="scl")  # synth: "tb", "anal", "resynth"; parse (only for synth="tb"): "std", "scl", "norm"
    #
    # synth_file(paths_tb_norm, instrument_detector=instrument_timbre_detector, refine_twm=True, pitch_shift=False,
    #            th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, create_tfrecords=False, synth="tb",
    #            parse="norm")  # synth: "tb", "anal", "resynth"; parse (only for synth="tb"): "std", "scl", "norm"

    #analyze_file(paths_tb, confidence_threshold=0.9, min_voiced_segment_ms=25)

    ########################
    # 1. Synthesize 10 trained Files with 3 Normalization Methods
    import pickle

    # Define the list of f_base_names
    f_base_names = [
        "03byLdNJnbcss00000187",
        "03KkVk8TYzluo00000164",
        "03TAoWzc23xZg00000184",
        "03un_J2R2G-c400000094",
        "04fdEVMZicC6A00000111",
        "04HHoYqjiRsAs00000119",
        "12qkXtU7vscRQ00000061",
        "13j2q2OX1jo1s00000099",
        "15k9ls6YgC7WY00000153",
        "19Df_oVHUiD5800040118"
    ]


    # Define paths dictionaries
    def generate_paths_dict(f_basename):
        paths_tb_std = {
            'original': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/original/{f_basename}.mp3',
            'f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/pitch_tracks/crepe/{f_basename}.f0.csv',

            'synth': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_std/resynth/{f_basename}.RESYN.wav',
            'synth_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_std/resynth/{f_basename}.RESYN.csv',
            'shifted': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_std/resynth/{f_basename}.shiftedRESYN.wav',
            'shifted_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_std/resynth/{f_basename}.shiftedRESYN.csv',

            'anal': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/anal/crepe/{f_basename}.npz',
            'timbre_cluster': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_std/{f_basename}.tb.csv'
        }

        paths_tb_scl = {
            'original': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/original/{f_basename}.mp3',
            'f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/pitch_tracks/crepe/{f_basename}.f0.csv',

            'synth': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/{f_basename}.RESYN.wav',
            'synth_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/{f_basename}.RESYN.csv',
            'shifted': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/{f_basename}.shiftedRESYN.wav',
            'shifted_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/{f_basename}.shiftedRESYN.csv',

            'anal': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/anal/crepe/{f_basename}.npz',
            'timbre_cluster': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/{f_basename}.tb.csv'
        }

        paths_tb_norm = {
            'original': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/original/{f_basename}.mp3',
            'f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/pitch_tracks/crepe/{f_basename}.f0.csv',

            'synth': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_normalized/resynth/{f_basename}.RESYN.wav',
            'synth_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_normalized/resynth/{f_basename}.RESYN.csv',
            'shifted': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_normalized/resynth/{f_basename}.shiftedRESYN.wav',
            'shifted_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_normalized/resynth/{f_basename}.shiftedRESYN.csv',

            'anal': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/anal/crepe/{f_basename}.npz',
            'timbre_cluster': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_normalized/{f_basename}.tb.csv'
        }

        return paths_tb_std, paths_tb_scl, paths_tb_norm


    # Load instrument timbre detector
    instrument_model_file = "/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/ic_100_0.05.pkl"
    with open(instrument_model_file, 'rb') as modelfile:
        instrument_timbre_detector = pickle.load(modelfile)


    # Function to synthesize files
    def synth_files(paths_dict, instrument_detector, synth="tb", parse="std"):
        for f_basename in f_base_names:
            paths_tb_std, paths_tb_scl, paths_tb_norm = generate_paths_dict(f_basename)

            if parse == "std":
                synth_file(paths_tb_std, instrument_detector=instrument_timbre_detector, refine_twm=True,
                           pitch_shift=False,
                           th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, create_tfrecords=False,
                           synth=synth,
                           parse=parse)
            elif parse == "scl":
                synth_file(paths_tb_scl, instrument_detector=instrument_timbre_detector, refine_twm=True,
                           pitch_shift=False,
                           th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, create_tfrecords=False,
                           synth=synth,
                           parse=parse)
            elif parse == "norm":
                synth_file(paths_tb_norm, instrument_detector=instrument_timbre_detector, refine_twm=True,
                           pitch_shift=False,
                           th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, create_tfrecords=False,
                           synth=synth,
                           parse=parse)


    # Call synth_files function for each set of paths
    #synth_files(paths_tb_std, instrument_timbre_detector, synth="tb", parse="std")
    #synth_files(paths_tb_scl, instrument_timbre_detector, synth="tb", parse="scl")
    #synth_files(paths_tb_norm, instrument_timbre_detector, synth="tb", parse="norm")


    # 2. Synthesize 2 Test Files with 3 Normalization Methods
    import pickle

    # Define the list of f_base_names
    f_base_names = [
        "15f1ORtWqE6sQ00000124",
        "1355vsSCpZ4tw00000217"
    ]


    # Define paths dictionaries
    def generate_paths_dict(f_basename):
        paths_tb_std = {
            'original': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/original/{f_basename}.mp3',
            'f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/pitch_tracks/crepe/{f_basename}.f0.csv',

            'synth': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_std/resynth/test/{f_basename}.RESYN.wav',
            'synth_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_std/resynth/test/{f_basename}.RESYN.csv',
            'shifted': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_std/resynth/test/{f_basename}.shiftedRESYN.wav',
            'shifted_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_std/resynth/test/{f_basename}.shiftedRESYN.csv',

            'anal': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/anal/crepe/{f_basename}.npz',
            'timbre_cluster': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_std/test/{f_basename}.tb.csv'
        }

        paths_tb_scl = {
            'original': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/original/{f_basename}.mp3',
            'f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/pitch_tracks/crepe/{f_basename}.f0.csv',

            'synth': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/test/{f_basename}.RESYN.wav',
            'synth_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/test/{f_basename}.RESYN.csv',
            'shifted': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/test/{f_basename}.shiftedRESYN.wav',
            'shifted_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/test/{f_basename}.shiftedRESYN.csv',

            'anal': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/anal/crepe/{f_basename}.npz',
            'timbre_cluster': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/test/{f_basename}.tb.csv'
        }

        paths_tb_norm = {
            'original': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/original/{f_basename}.mp3',
            'f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/pitch_tracks/crepe/{f_basename}.f0.csv',

            'synth': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_normalized/resynth/test/{f_basename}.RESYN.wav',
            'synth_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_normalized/resynth/test/{f_basename}.RESYN.csv',
            'shifted': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_normalized/resynth/test/{f_basename}.shiftedRESYN.wav',
            'shifted_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_normalized/resynth/test/{f_basename}.shiftedRESYN.csv',

            'anal': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/anal/crepe/{f_basename}.npz',
            'timbre_cluster': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_normalized/test/{f_basename}.tb.csv'
        }

        return paths_tb_std, paths_tb_scl, paths_tb_norm


    # Load instrument timbre detector
    instrument_model_file = "/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test/ic_100_0.05.pkl"
    with open(instrument_model_file, 'rb') as modelfile:
        instrument_timbre_detector = pickle.load(modelfile)


    # Function to synthesize files
    def synth_files(paths_dict, instrument_detector, synth="tb", parse="std", hmag_std_scl=None, inharm=False):
        for f_basename in f_base_names:
            paths_tb_std, paths_tb_scl, paths_tb_norm = generate_paths_dict(f_basename)

            if parse == "std":
                synth_file(paths_tb_std, instrument_detector=instrument_timbre_detector, refine_twm=True,
                           pitch_shift=False,
                           th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, create_tfrecords=False,
                           synth=synth,
                           parse=parse)
            elif parse == "scl":
                synth_file(paths_tb_scl, instrument_detector=instrument_timbre_detector, refine_twm=True,
                           pitch_shift=False,
                           th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, create_tfrecords=False,
                           synth=synth, parse=parse, hmag_std_scl=hmag_std_scl, inharm=inharm)
            elif parse == "norm":
                synth_file(paths_tb_norm, instrument_detector=instrument_timbre_detector, refine_twm=True,
                           pitch_shift=False,
                           th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, create_tfrecords=False,
                           synth=synth,
                           parse=parse)


    # Call synth_files function for each set of paths
    # synth_files(paths_tb_std, instrument_timbre_detector, synth="tb", parse="std")
    # synth_files(paths_tb_scl, instrument_timbre_detector, synth="tb", parse="scl")
    # synth_files(paths_tb_norm, instrument_timbre_detector, synth="tb", parse="norm")


    # 3. Synthesize with harmonic magnitude deviation from Gaussian
    ## 3.1 Training Set
    # TODO: Model Inharmonicity - Gaussian sampled from training data
    f_base_names = [
        "03byLdNJnbcss00000187",
        "03KkVk8TYzluo00000164",
        "03TAoWzc23xZg00000184",
        "03un_J2R2G-c400000094",
        "04fdEVMZicC6A00000111",
        "04HHoYqjiRsAs00000119",
        "12qkXtU7vscRQ00000061",
        "13j2q2OX1jo1s00000099",
        "15k9ls6YgC7WY00000153",
        "19Df_oVHUiD5800040118"
    ]
    def generate_paths_dict_train(f_basename):

        paths_tb_scl_hmag_std_0_1 = {
            'original': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/original/{f_basename}.mp3',
            'f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/pitch_tracks/crepe/{f_basename}.f0.csv',

            'synth': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/hmag_std_0.1/{f_basename}.RESYN.wav',
            'synth_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/hmag_std_0.1/{f_basename}.RESYN.csv',
            'shifted': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/hmag_std_0.1/{f_basename}.shiftedRESYN.wav',
            'shifted_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/hmag_std_0.1/{f_basename}.shiftedRESYN.csv',

            'anal': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/anal/crepe/{f_basename}.npz',
            'timbre_cluster': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/{f_basename}.tb.csv'
        }

        paths_tb_scl_hmag_std_0_05 = {
            'original': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/original/{f_basename}.mp3',
            'f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/pitch_tracks/crepe/{f_basename}.f0.csv',

            'synth': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/hmag_std_0.05/{f_basename}.RESYN.wav',
            'synth_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/hmag_std_0.05/{f_basename}.RESYN.csv',
            'shifted': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/hmag_std_0.05/{f_basename}.shiftedRESYN.wav',
            'shifted_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/hmag_std_0.05/{f_basename}.shiftedRESYN.csv',

            'anal': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/anal/crepe/{f_basename}.npz',
            'timbre_cluster': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/{f_basename}.tb.csv'
        }

        return paths_tb_scl_hmag_std_0_1, paths_tb_scl_hmag_std_0_05

    # def synth_files_harm_std(instrument_detector, synth="tb", parse="std", hmag_std_scl=None, inharm=None):
    #     for f_basename in f_base_names:
    #         paths_tb_scl_hmag_std = generate_paths_dict(f_basename)
    #
    #         if parse == "scl":
    #             synth_file(paths_tb_scl_hmag_std, instrument_detector=instrument_detector, refine_twm=True,
    #                        pitch_shift=False,
    #                        th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, create_tfrecords=False,
    #                        synth=synth, parse=parse, hmag_std_scl=hmag_std_scl, inharm=inharm)

    def synth_files_harm_std(instrument_detector, synth="tb", parse="std", hmag_std_scl=None, inharm=None):
        for f_basename in f_base_names:
            paths_tb_scl_hmag_std_0_1, paths_tb_scl_hmag_std_0_05 = generate_paths_dict_train(f_basename)

            if parse == "scl":
                if hmag_std_scl == 0.1:
                    paths_tb_scl_hmag_std = paths_tb_scl_hmag_std_0_1
                elif hmag_std_scl == 0.05:
                    paths_tb_scl_hmag_std = paths_tb_scl_hmag_std_0_05

                synth_file(paths_tb_scl_hmag_std, instrument_detector=instrument_detector, refine_twm=True,
                           pitch_shift=False,
                           th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, create_tfrecords=False,
                           synth=synth, parse=parse, hmag_std_scl=hmag_std_scl, inharm=inharm)

    # print("synthesizing training data hmag_std_scl=0.1...")
    # synth_files_harm_std(instrument_timbre_detector, synth="tb", parse="scl", hmag_std_scl=0.1, inharm=None)
    # print("synthesizing training data hmag_std_scl=0.05...")
    # synth_files_harm_std(instrument_timbre_detector, synth="tb", parse="scl", hmag_std_scl=0.05, inharm=None)

    ## 3.2 Test Set
    f_base_names = [
        "15f1ORtWqE6sQ00000124",
        "1355vsSCpZ4tw00000217"
    ]


    def generate_paths_dict(f_basename):

        paths_tb_scl_hmag_std_0_1 = {
            'original': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/original/{f_basename}.mp3',
            'f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/pitch_tracks/crepe/{f_basename}.f0.csv',

            'synth': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/test/hmag_std_0.1/{f_basename}.RESYN.wav',
            'synth_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/test/hmag_std_0.1/{f_basename}.RESYN.csv',
            'shifted': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/test/hmag_std_0.1/{f_basename}.shiftedRESYN.wav',
            'shifted_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/test/hmag_std_0.1/{f_basename}.shiftedRESYN.csv',

            'anal': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/anal/crepe/{f_basename}.npz',
            'timbre_cluster': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/test/{f_basename}.tb.csv'
        }

        paths_tb_scl_hmag_std_0_05 = {
            'original': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/original/{f_basename}.mp3',
            'f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/pitch_tracks/crepe/{f_basename}.f0.csv',

            'synth': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/test/hmag_std_0.05/{f_basename}.RESYN.wav',
            'synth_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/test/hmag_std_0.05/{f_basename}.RESYN.csv',
            'shifted': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/test/hmag_std_0.05/{f_basename}.shiftedRESYN.wav',
            'shifted_f0': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/resynth/test/hmag_std_0.05/{f_basename}.shiftedRESYN.csv',

            'anal': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo/anal/crepe/{f_basename}.npz',
            'timbre_cluster': f'/Users/qinliu/Postgraduate/Master/MTG-Master_Thesis/DDSP/constrained-harmonic-resynthesis/timbre_cluster_minmax/test/{f_basename}.tb.csv'
        }

        return paths_tb_scl_hmag_std_0_1, paths_tb_scl_hmag_std_0_05


    def synth_files_harm_std(instrument_detector, synth="tb", parse="std", hmag_std_scl=None, inharm=None):
        for f_basename in f_base_names:
            paths_tb_scl_hmag_std_0_1, paths_tb_scl_hmag_std_0_05 = generate_paths_dict(f_basename)

            if parse == "scl":
                if hmag_std_scl == 0.1:
                    paths_tb_scl_hmag_std = paths_tb_scl_hmag_std_0_1
                elif hmag_std_scl == 0.05:
                    paths_tb_scl_hmag_std = paths_tb_scl_hmag_std_0_05

                synth_file(paths_tb_scl_hmag_std, instrument_detector=instrument_detector, refine_twm=True,
                           pitch_shift=False,
                           th_lc=0.2, th_hc=0.7, voiced_th_ms=100, sawtooth_synth=False, create_tfrecords=False,
                           synth=synth, parse=parse, hmag_std_scl=hmag_std_scl, inharm=inharm)


    # print("synthesizing test data hmag_std_scl=0.1...")
    # synth_files_harm_std(instrument_timbre_detector, synth="tb", parse="scl", hmag_std_scl=0.1, inharm=None)
    # print("synthesizing test data hmag_std_scl=0.05...")
    # synth_files_harm_std(instrument_timbre_detector, synth="tb", parse="scl", hmag_std_scl=0.05, inharm=None)

    f_basename_anal = "1355vsSCpZ4tw00000217"
    paths_tb_anal = {
        'original': f'../ViolinEtudes-zenodo-test/original/{f_basename}.mp3',
        'f0': f'../ViolinEtudes-zenodo-test/pitch_tracks/crepe/{f_basename}.f0.csv',

        'synth': f'../ViolinEtudes-zenodo-test/resynth/ic_100_0.05_crepe/timbre_cluster_synthesis_n_harm=25/{f_basename}.RESYN.wav',
        'synth_f0': f'../ViolinEtudes-zenodo-test/resynth/ic_100_0.05_crepe/timbre_cluster_synthesis_n_harm=25/{f_basename}.RESYN.csv',

        'anal': f'../ViolinEtudes-zenodo-test/anal/time_test/{f_basename}.npz',
        'timbre_cluster': f'../ViolinEtudes-zenodo-test/timber_cluster_88/{f_basename}.tb.csv'
    }
    print("analyzing for timestamp test...")
    #analyze_file(paths_tb, confidence_threshold=0.6, min_voiced_segment_ms=25)