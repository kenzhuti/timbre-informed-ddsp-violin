import copy
from scipy.signal import medfilt
import numpy as np
import sys
import os

HOP_SIZE = 128
SAMPLING_RATE = 44100
sms_tools_path = 'sms-tools'  # if you use git clone to home dir (e.g. in colab)
# change according to the install location of https://github.com/MTG/sms-tools

sys.path.append(os.path.join(sms_tools_path, 'software', 'models',
                             './utilFunctions_C/'))
try:
    import utilFunctions_C as UF_C
except ImportError:
    print("\n")
    print("--------------------------------------------------------------------")
    print("Warning:")
    print("Cython modules for some of the core functions were not imported.")
    print("Please refer to the README.md file in the 'sms-tools' directory,")
    print("for the instructions to compile the cython modules.")
    print("Exiting the code!!")
    print("--------------------------------------------------------------------")
    print("\n")
    sys.exit(0)


def silence_segments_one_run(confidences, confidence_threshold, segment_len_th):
    conf_bool = np.array(confidences > confidence_threshold).reshape(-1)
    absdiff = np.abs(np.diff(np.concatenate(([False], conf_bool, [False]))))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    segment_durs = np.diff(ranges, axis=1)
    valid_segments = ranges[np.repeat(segment_durs > segment_len_th, repeats=2, axis=1)].reshape(-1, 2)
    voiced = np.zeros(len(confidences), dtype=bool)
    for segment in valid_segments:
        voiced[segment[0]:segment[1]] = True
    return voiced


def silence_unvoiced_segments(pitch_track_csv, low_confidence_threshold=0.2,
                              high_confidence_threshold=0.7, min_voiced_segment_ms=12):
    """
    Accepts crepe (or any other pitch estimator) output in the csv format and removes unvoiced segments based on two confidence thresholds.
    Removes pitch estimates with confidences below the low threshold. Accepts estimates with confidences above the high threshold.
    If the confidence is in between high and low values, then apply a median filtering with the voiced_segment_ms parameter.
    The same segment duration is also used to silence short voiced segments, which sound like arbitrary noises.
    :param pitch_track_csv: csv with ["time", "frequency", "confidence"] columns
    :param low_confidence_threshold: confidence threshold in range (0,1).
    :param high_confidence_threshold: confidence threshold in range (0,1)/
    :param min_voiced_segment_ms: voiced segments shorter than the specified lenght are discarded.
    :return: input csv file with the silenced segments
    """
    annotation_interval_ms = 1000 * pitch_track_csv.loc[:1, "time"].diff()[1]
    voiced_th = int(np.ceil(min_voiced_segment_ms / annotation_interval_ms))

    # we do not accept the segment if a close neighbors do not have a confidence > 0.7
    smoothened_confidences = medfilt(pitch_track_csv["confidence"], kernel_size=2 * (voiced_th // 2) + 1)
    smooth_voiced = silence_segments_one_run(smoothened_confidences,
                                             confidence_threshold=high_confidence_threshold, segment_len_th=voiced_th)

    # we also do not accept the pitch values if the individual confidences are really low
    hard_voiced = silence_segments_one_run(pitch_track_csv["confidence"],
                                           confidence_threshold=low_confidence_threshold, segment_len_th=voiced_th)

    # we accept the intersection of these two zones
    voiced = np.logical_and(smooth_voiced, hard_voiced)

    smoothened_pitch = copy.deepcopy(pitch_track_csv["frequency"])
    smoothened_pitch[~voiced] = np.nan
    smoothened_pitch.fillna(smoothened_pitch.rolling(window=15, min_periods=8).median(), inplace=True)

    # medfilt(pitch_track_csv["frequency"], kernel_size=21)
    absdiff = np.abs(np.diff(np.concatenate(([False], voiced, [False]))))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    unvoiced_ranges = np.vstack([ranges[:-1, 1], ranges[1:, 0]]).T
    for unvoiced_boundary in unvoiced_ranges:
        # we don't want small unvoiced zones. Check if they are acceptable with a more favorable mean thresholding
        len_unvoiced = np.diff(unvoiced_boundary)[0]
        if len_unvoiced < voiced_th:
            avg_confidence = pitch_track_csv.loc[unvoiced_boundary[0]:unvoiced_boundary[1], "confidence"].mean()
            if avg_confidence > low_confidence_threshold:
                voiced[unvoiced_boundary[0]:unvoiced_boundary[1]] = True
            elif len_unvoiced < 8:
                # and (unvoiced_boundary[0] > 3) and (unvoiced_boundary[-1] < len(pitch_track_csv)-3):
                pitch_track_csv.loc[unvoiced_boundary[0]:unvoiced_boundary[1], "frequency"] = \
                    smoothened_pitch[unvoiced_boundary[0]:unvoiced_boundary[1]]
                voiced[unvoiced_boundary[0]:unvoiced_boundary[1]] = True

    pitch_track_csv.loc[~voiced, "frequency"] = 0
    pitch_track_csv["frequency"] = pitch_track_csv["frequency"].fillna(0)
    return pitch_track_csv


def refinef0Twm(pfreq, pmag, f0c, refinement_range_cents=10):
    """
    Function to refine the f0 estimate with specified cents interval around the original pitch track. Data-driven f0 estimation methods
    are good against octave errors and noise. However, they can introduce bias towards the data labels (e.g. western tuning system)
    if we do a fine-grained analysis. The purpose of this refinement step is to combine the best of both worlds in a post-processing
    step, using the sinusoids that we have a direct access.

    pfreq, pmag: peak frequencies and magnitudes in the analysis window,
    f0c: the f0 candidate provided by the pitch_tracker (float)
    refinement_range_cents: how many cents we are allowed to deviate for the new f0 prediction (10c deviation -> pick from 21 candidates)
    returns f0: the refined fundamental frequency in Hz
    """
    f0c = f0c * np.power(2, (np.array(range(-refinement_range_cents, 1 + refinement_range_cents)) / 1200))

    f0, f0error = UF_C.twm(pfreq, pmag, f0c)  # call the TWM function with peak candidates

    if f0 > 0:  # accept and return f0 if below max error allowed
        return f0, f0error
    else:
        return 0, np.inf


def refine_harmonics_twm(hfreq, hmag, hphases, f0, f0et=5.0, f0_refinement_range_cents=10, min_voiced_segment_ms=100):
    """
    Refine the f0 estimate with the help of two-way mismatch algorithm and change the harmonic components
    to the exact multiples of the refined f0 estimate
    :param hfreq: analyzed harmonic frequencies
    :param hmag: analyzed magnitudes
    :param f0: f0 in Hz before TWM
    :param f0et: error threshold for the TWM
    :param f0_refinement_range_cents: the range to be explored in TWM
    :return: new synthesis parameters
    """
    for frame, f0_frame in enumerate(f0):
        if f0_frame > 0:  # for the valid frequencies
            pfreq = hfreq[frame]
            pmag = hmag[frame]
            f0_twm, f0err_twm = refinef0Twm(pfreq, pmag, f0_frame, refinement_range_cents=f0_refinement_range_cents)
            if f0err_twm < f0et:
                hfreq[frame] = f0_twm * np.round(pfreq / f0_twm)
                f0[frame] = f0_twm
            else:
                f0[frame] = 0
                hfreq[frame] = 0
                hmag[frame] = -100

    min_voiced_segment_len = int(np.ceil((min_voiced_segment_ms / 1000) / (HOP_SIZE / SAMPLING_RATE)))
    voiced = silence_segments_one_run(f0, 0, min_voiced_segment_len)
    f0[~voiced] = 0
    hfreq[~voiced] = 0
    hmag[~voiced] = -100
    if hphases.size != 0:
        hphases[~voiced] = 0
    return hfreq, hmag, hphases, f0


def supress_timbre_anomalies(instrument_detector, hfreq, hmag, hphase, f0):
    hmag_ptr = np.copy(hmag)[:, 1:25] # # 12->25 for violin
    hmag_ptr[np.isnan(hmag_ptr)] = -100
    hvalid = np.logical_and(hmag_ptr[:, 0] > -100, f0 > 0)
    hmag_ptr[hvalid] = hmag_ptr[hvalid] - hmag[hvalid, 0][:, np.newaxis]
    filter_centroids = np.array(list(instrument_detector.keys()))
    f0_classes = np.abs(f0[:, np.newaxis] - filter_centroids[np.newaxis, :]).argmin(axis=1)
    relevant_classes = np.unique(f0_classes[hvalid])
    voiced = hvalid
    for pitch_class in relevant_classes:
        relevant_timbre_model = instrument_detector[filter_centroids[pitch_class]]
        relevant_section = np.logical_and(f0_classes == pitch_class, hvalid)
        voiced_relevant = relevant_timbre_model.predict(hmag_ptr[relevant_section])
        voiced[relevant_section] = voiced_relevant > 0

    f0[~voiced] = 0
    hfreq[~voiced] = 0
    hmag[~voiced] = -100
    if hphase.size != 0:
        hphase[~voiced] = 0
    return hfreq, hmag, hphase, f0

