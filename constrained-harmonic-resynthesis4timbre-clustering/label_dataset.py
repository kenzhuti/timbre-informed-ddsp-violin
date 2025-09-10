import numpy as np
import os
from time import time as taymit
from joblib import Parallel, delayed
import glob
import pickle
from sklearn.covariance import EllipticEnvelope
from constrained_harmonic_resynthesis import analyze_file, synth_file
import argparse


def gen_paths(main_path: object, modeln: object, suffix: object, pitch_shift: object = False,
              use_anal_paths: object = False, create_tfrecords: object = False) -> object:
    # TODO: glob enable nested folders, mp3 and wav all possibilities
    originals = sorted(glob.glob(os.path.join(main_path, 'original', '**')))
    paths = []
    for i, original in enumerate(originals):
        name = os.path.relpath(original, os.path.join(main_path, "original"))[:-4]
        f0 = os.path.join(main_path, 'pitch_tracks_finetuned', modeln, name + '.f0.csv') #'pitch_tracks_chr'
        resynth_path = os.path.join(main_path, 'resynth', suffix, name)
        if not os.path.exists(os.path.dirname(resynth_path)):
            os.makedirs(os.path.dirname(resynth_path))
        p = {
            'original': original,
            'f0': f0,
            'synth': resynth_path + '.RESYN.wav',
            'synth_f0': resynth_path + '.RESYN.csv'
        }
        if pitch_shift:
            p['shifted'] = resynth_path + '.shiftedRESYN.wav'
            p['shifted_f0'] = resynth_path + '.shiftedRESYN.csv'

        if create_tfrecords:
            tfrecord_path = os.path.join(main_path, 'tfrecord', suffix, name)
            if not os.path.exists(os.path.dirname(tfrecord_path)):
                os.makedirs(os.path.dirname(tfrecord_path))
            p["synth_tfrecord"] = tfrecord_path + '.RESYN.tfrecord'
            if pitch_shift:
                p['shifted_tfrecord'] = tfrecord_path + '.shiftedRESYN.tfrecord'
        if use_anal_paths:
            anal_path = os.path.join(main_path, 'anal', modeln, name + '.npz')
            if not os.path.exists(os.path.dirname(anal_path)):
                os.makedirs(os.path.dirname(anal_path))
            p['anal'] = anal_path
        paths.append(p)
    return paths


def chr_dataset(dataset_folder=os.path.join(os.path.expanduser("~"), "FluteEtudes"),
                model="finetuned_instrument_model_100_005",
                iteration_if_applicable=0,  # do not use if starting from the original model,
                hcc=False, #True,  # whether to use harmonic consistency constraint (HCC)
                ic=True,  # whether to use instrument-modeling constraint (IC). if not, only use HCC.
                ic_new_model=True,  # whether to estimate a new instrument model.
                # only applicable if ic=True. if false, use the stored instrument model
                ic_use_existing_anal_files=False,  # if we have stored the harmonic analysis output, use them for ic.
                low_confidence_threshold=0.0, #0.3,
                high_confidence_threshold=0.7,
                min_voiced_th_ms=50,
                synthesize_pitch_shifted_versions=True,
                use_sawtooth_timbre=False,
                create_tfrecords=False,
                n_parallel_jobs=16):
    """
    Apply Constrained Harmonic Resynthesis (one iteration of silencing) to a monophonic single instrument dataset with
    automatically extracted pitch labels. 1) apply a filtering based on confidences given by the pitch tracker.
    The relevant parameters for this step are low_confidence_threshold, high_confidence_threshold, and min_voiced_th_ms.
    2) silence pitch tracks with anomalous harmonics for this specific instrument. The relevant parameters are ic(_xxx)
    3) apply harmonic consistency constraints with two-way mismatch procedure, and set the harmonics at exact multiples.
    The algorithm will retain the folder structure and the naming conventions of the original dataset, and output
    resynthesized and tfrecord versions (for re-training crepe).
    (.wav) and f0 (.f0.csv) files.
    :param dataset_folder: the main folder of the unlabeled monophonic, single instrument dataset. It should have the
    subfolders 1)original: where the original .mp3 or .wav files are stored, and 2)f0s/model: where the f0 tracks (.csv)
    extracted with the specified model are stored.
    :param model: a given name for the extracted f0s path (e.g., crepe, tape, finetuned_iter2 etc.
    :param iteration_if_applicable: If n, save the data with n+1 namings.
    :param hcc: Harmonic consistency constraint. default=True
    :param ic: Instrument-modeling constraint. default=True
    :param ic_new_model: Generate new instrument model (elliptic envelope anomaly detector) from data. default=True
    :param ic_use_existing_anal_files: default=False. In case of pre-computed analysis files, use them.
    :param low_confidence_threshold: default 0.3 in (0-1) -> remove pitch estimates with confidences below the low threshold
    :param high_confidence_threshold: default 0.7 in (0,1) -> accept pitch estimates with very high confidence
    :param min_voiced_th_ms: for in-between values, apply a median filter guided by confidences
    :param synthesize_pitch_shifted_versions: pitch shifted versions make the data less vulnerable to the auto-tuning ef
    :param use_sawtooth_timbre: use same harmonic structure as an ablation study
    :param n_parallel_jobs: SMS takes very long to run!! Use parallelization!
    :return: write folders synth and tfrecord (also microtonal pitch shifted versions)
    """

    refine_estimates_with_twm = hcc #hcc  # the descriptive old name for Harmonic Consistency Constraint

    if ic:  # use instrument-modeling constraint
        num_filters = 100
        contamination = 0.05# 0.05
        name_suffix = "ic_" + str(num_filters) + '_' + str(contamination)
    else:  # only use harmonic consistency constraint
        instrument_timbre_detector = None
        name_suffix = "standard"
    if use_sawtooth_timbre:
        name_suffix = name_suffix + "sawtooth"
    if iteration_if_applicable:
        name_suffix = name_suffix + "_iter" + str(iteration_if_applicable)
    name_suffix = name_suffix + "_" + model

    if ic and ic_new_model:  # create paths for the analysis files (SMS tools is slow, don't estimate it every run)
        use_anal_paths = True
    else:
        use_anal_paths = False

    paths_list = gen_paths(dataset_folder, modeln=model, suffix=name_suffix, pitch_shift=synthesize_pitch_shifted_versions,
                           use_anal_paths=use_anal_paths, create_tfrecords=create_tfrecords)
    if ic:
        # Instrument model is used for the standard implementation, below is the code to create the
        # instrument timbre model
        if ic_new_model:
            print("started instrument model estimation")
            # combine instrument model estimation with the synthesis. The analysis for the instrument estimation takes
            # a long while, so only do it when really needed!
            if not ic_use_existing_anal_files:
                #analyze_file(paths_list[0], confidence_threshold=0.7, min_voiced_segment_ms=30)
                Parallel(n_jobs=n_parallel_jobs)(delayed(analyze_file)(
                    paths_dict, confidence_threshold=0.2) for paths_dict in paths_list)

            data, pitch_content = [], []
            for paths_dict in paths_list:
                file_content = np.load(paths_dict['anal'])
                data.append(file_content['hmag'])
                pitch_content.append(file_content['f0'])

            data = np.vstack(data)
            pitch_content = np.hstack(pitch_content)
            order = pitch_content.argsort()
            pitch_content = pitch_content[order]
            data = data[order]
            data = data - data[:, 0][:, np.newaxis]

            mids = np.linspace(pitch_content[10], pitch_content[-11], num_filters + 2)  # not directly the max
            mids[0] = pitch_content[0]         # a quick fix for outliers, to make sure that we do not focus
            mids[-1] = pitch_content[-1]       # on some arbitrary big or small pitch estimate
            instrument_timbre_detectors = {}
            print('range:', min(mids), max(mids))
            for n in range(1, num_filters + 1):
                mid = mids[n]
                start = mids[n-1]
                end = mids[n+1]
                relevant = np.logical_and(pitch_content > start, pitch_content < end)
                relevant_data = data[relevant]
                if len(relevant_data>10):
                    instrument_timbre_detector = EllipticEnvelope(contamination=contamination).fit(relevant_data[:, 1:])
                    instrument_timbre_detectors[mid] = instrument_timbre_detector
                    print(mid, n, start, end, len(relevant_data))
                    print(np.array2string(instrument_timbre_detector.location_, precision=2))
                else:
                    print('not enough samples to generate instrument model in the range:', start, end)
                    print('skipping')

            pitch_bins = mids[1:-2]
            pitch_hist, _ = np.histogram(pitch_content, bins=pitch_bins)
            pitch_dist = pitch_hist / len(data)

            print("Pitch distribution for the instrument model:")
            for f, p in zip(pitch_bins, pitch_dist):
                print("%4d" % f, "*" * int(p * 100))

            with open(os.path.join(dataset_folder, 'ic_' + str(num_filters) + '_' +
                                                   str(contamination) + '.pkl'), 'wb') as outp:
                pickle.dump(instrument_timbre_detectors, outp, pickle.HIGHEST_PROTOCOL)
            print("FINISHED INSTRUMENT MODEL ESTIMATION!!! \n\n\n\n\n\n\n\n NOW THE SYNTHESIS STARTS!!!")

        instrument_model_file = os.path.join(dataset_folder, 'ic_' + str(num_filters) + '_' +
                                             str(contamination) + '.pkl')
        with open(instrument_model_file, 'rb') as modelfile:
            instrument_timbre_detector = pickle.load(modelfile)

    time_grade = taymit()
    Parallel(n_jobs=n_parallel_jobs)(delayed(synth_file)(
        paths_dict, pitch_shift=synthesize_pitch_shifted_versions,
        instrument_detector=instrument_timbre_detector,
        th_lc=low_confidence_threshold, th_hc=high_confidence_threshold, voiced_th_ms=min_voiced_th_ms,
        refine_twm=refine_estimates_with_twm, sawtooth_synth=use_sawtooth_timbre, create_tfrecords=create_tfrecords
    ) for paths_dict in paths_list)

    time_grade = taymit() - time_grade
    print("It took {:.3f}".format(time_grade))


def arg_parser():
    parser = argparse.ArgumentParser('ConstrainedHarmonicResynthesizer', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-path', '-p',
                        #default=os.path.join(os.path.expanduser("~"), "musical-etudes", "clarinet-etudes"),
                        default=os.path.join(os.path.expanduser("~"), "Postgraduate/Master/MTG-Master_Thesis/DDSP/flute-etudes"),
                        help='single instrument  dataset path which contains the "original" folder')
    parser.add_argument('--create-tfrecords', '-t', default=False,
                        help='Create tfrecord files to directly finetune the CREPE')
    parser.add_argument('--model', '-m', help='pitch estimation model for harmonic reconstruction', default="crepe")
    parser.add_argument('--instrument-constraint', '-ic', help='whether to model the instrument constraints for single timbre dataset',
                        default=True)
    parser.add_argument('--pitch-shifts', '-ps', default='False', #default='True',
                        help='Whether to apply microtonal pitch shifts')
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = arg_parser()

    # args['dataset_path'] = os.path.join(os.path.expanduser("~"), "Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-test_2025")
    # args['dataset_path'] = "/Volumes/Backup Plus/Dataset_mobile_HDD/ViolinEtudes-zenodo"
    args['instrument_constraint'] = True
    args['pitch_shifts'] = False
    args['create_tfrecords'] = False
    args['model'] = "crepe" #"finetuned_violin_model_100_010"
    args['instrument_constraint'] = True

    # 1 Downsample, ic_use_existing_anal_files=True: synthesize from existing .anal files;
    # ic_new_model=False: Generate new instrument model (elliptic envelope anomaly detector) from data from analyze_file()
    args['dataset_path'] = os.path.join(os.path.expanduser("~"),
                                        "Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo-downsample_2025/train")
    chr_dataset(dataset_folder=args["dataset_path"], model=args["model"], ic=args["instrument_constraint"],
                synthesize_pitch_shifted_versions=args["pitch_shifts"], create_tfrecords=args["create_tfrecords"],
                ic_use_existing_anal_files=True, ic_new_model=True) # 2

    # 2 All anal without synth
    # args['dataset_path'] = os.path.join(os.path.expanduser("~"),
    #                                     "Postgraduate/Master/MTG-Master_Thesis/DDSP/ViolinEtudes-zenodo_2025")
    # chr_dataset(dataset_folder=args["dataset_path"], model=args["model"], ic=args["instrument_constraint"],
    #             synthesize_pitch_shifted_versions=args["pitch_shifts"], create_tfrecords=args["create_tfrecords"],
    #             ic_use_existing_anal_files=False, ic_new_model=True) # 1
