import logging
import re
import numpy as np
import glob
import os.path
import torch

import mne

log = logging.getLogger(__name__)

def session_key(file_name):
    """ Sort the file name by session """
    return re.findall(r'(s\d{2})', file_name)

def natural_key(file_name):
    """ Provides a human-like sorting key of a string """
    key = [int(token) if token.isdigit() else None
           for token in re.split(r'(\d+)', file_name)]
    return key

def time_key(file_name):
    """ Provides a time-based sorting key """
    splits = file_name.split('/')
    [date] = re.findall(r'(\d{4}_\d{2}_\d{2})', splits[-2])
    date_id = [int(token) for token in date.split('_')]
    recording_id = natural_key(splits[-1])
    session_id = session_key(splits[-2])

    return date_id + session_id + recording_id

def read_all_file_names(path, extension, key="time"):
    """ Read all files with specified extension from given path """
    file_paths = glob.glob(path + '**/*' + extension, recursive=True)
    return file_paths  # Returning without sorting

    if key == 'time':
        return sorted(file_paths, key=time_key)
    elif key == 'natural':
        return sorted(file_paths, key=natural_key)

def get_info_with_mne(file_path):
    """ Read info from the EDF file without loading the data """
    try:
        if 'eval' in file_path:
            edf_file = mne.io.read_raw_edf(file_path, eog=['FP1', 'FP2', 'F3', 'F4',
                                                           'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                                                           'T3', 'T4', 'T5', 'T6', 'PZ', 'FZ', 'CZ', 'A1', 'A2'], verbose='error')
        else:
            edf_file = mne.io.read_raw_edf(file_path, verbose='error')
    except ValueError:
        return None, None, None, None, None, None

    sampling_frequency = int(edf_file.info['sfreq'])
    if sampling_frequency < 10:
        sampling_frequency = 1 / (edf_file.times[1] - edf_file.times[0])
        if sampling_frequency < 10:
            return None, sampling_frequency, None, None, None, None

    n_samples = edf_file.n_times
    signal_names = edf_file.ch_names
    n_signals = len(signal_names)
    duration = n_samples / max(sampling_frequency, 1)

    return edf_file, sampling_frequency, n_samples, n_signals, signal_names, duration

def get_recording_length(file_path):
    """ Open the EDF as bytes and only read the header to parse the duration """
    with open(file_path, 'rb') as f:
        header = f.read(256)
    return int(header[236:244].decode('ascii'))

def load_data(fname, preproc_functions, sensor_types=['EEG']):
    cnt, sfreq, n_samples, n_channels, chan_names, n_sec = get_info_with_mne(fname)
    log.info("Load data..." + fname)

    # Set GPU device
    torch.cuda.set_device(0)

    cnt.load_data()
    selected_ch_names = []
    if 'EEG' in sensor_types:
        wanted_elecs = ['A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1',
                        'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']

        for wanted_part in wanted_elecs:
            wanted_found_name = []
            for ch_name in cnt.ch_names:
                if (' ' + wanted_part + '-' in ch_name) or (wanted_part == ch_name):
                    wanted_found_name.append(ch_name)
            print(wanted_found_name)  # Comment out if not needed
            assert len(wanted_found_name) == 1
            selected_ch_names.append(wanted_found_name[0])
    if 'EKG' in sensor_types:
        wanted_found_name = []
        for ch_name in cnt.ch_names:
            if 'EKG' in ch_name:
                wanted_found_name.append(ch_name)
        assert len(wanted_found_name) == 1
        selected_ch_names.append(wanted_found_name[0])

    cnt = cnt.pick_channels(selected_ch_names)

    n_sensors = 0
    if 'EEG' in sensor_types:
        n_sensors += 21
    if 'EKG' in sensor_types:
        n_sensors += 1

    assert len(cnt.ch_names) == n_sensors, (
        "Expected {:d} channel names, got {:d} channel names".format(
            n_sensors, len(cnt.ch_names)))

    # Change from volt to microvolt
    data = (cnt.get_data() * 1e6).astype(np.float32)
    fs = cnt.info['sfreq']
    log.info("Preprocessing...")
    if data.shape[1] < 120000:
        return None
    for fn in preproc_functions:
        log.info(fn)
        print(data.shape)
        data, fs = fn(data, fs)
        data = data.astype(np.float32)
        fs = float(fs)

    return data

def get_all_sorted_file_names_and_labels(train_or_eval, folders):
    all_file_names = []
    for folder in folders:
        full_folder = os.path.join(folder, train_or_eval) + '/'
        log.info("Reading {:s}...".format(full_folder))
        this_file_names = read_all_file_names(full_folder, '.edf', key='time')
        log.info(".. {:d} files.".format(len(this_file_names)))
        all_file_names.extend(this_file_names)
    log.info("{:d} files in total.".format(len(all_file_names)))

    labels = ['/abnormal/' in f for f in all_file_names]
    labels = np.array(labels).astype(np.int64)
    return all_file_names, labels

class DiagnosisSet(object):
    def __init__(self, n_recordings, max_recording_mins, preproc_functions,
                 data_folders,
                 train_or_eval='train', sensor_types=['EEG'],):
        self.n_recordings = n_recordings
        self.max_recording_mins = max_recording_mins
        self.preproc_functions = preproc_functions
        self.train_or_eval = train_or_eval
        self.sensor_types = sensor_types
        self.data_folders = data_folders

    def load(self, only_return_labels=False, batch_size=256):
        log.info("Read file names")
        all_file_names, labels = get_all_sorted_file_names_and_labels(
            train_or_eval=self.train_or_eval,
            folders=self.data_folders,)

        if self.max_recording_mins is not None:
            log.info("Read recording lengths...")
            assert 'train' == self.train_or_eval

            lengths = [get_recording_length(fname) for fname in all_file_names]
            lengths = np.array(lengths)
            mask = lengths < self.max_recording_mins * 60
            cleaned_file_names = np.array(all_file_names)[mask]
            cleaned_labels = labels[mask]
        else:
            cleaned_file_names = np.array(all_file_names)
            cleaned_labels = labels
        if only_return_labels:
            return cleaned_labels
        X = []
        y = []
        n_files = len(cleaned_file_names[:self.n_recordings])
        for i_fname, fname in enumerate(cleaned_file_names[:self.n_recordings]):
            log.info("Load {:d} of {:d}".format(i_fname + 1, n_files))
            x = load_data(fname, preproc_functions=self.preproc_functions,
                          sensor_types=self.sensor_types)
            if x is None:
                continue
            X.append(x)
            y.append(cleaned_labels[i_fname])
        y = np.array(y)
        return X, y
