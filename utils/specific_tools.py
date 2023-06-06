'''
Written by: Saksham Consul 04/28/2023
Scripts needed for specific application
'''
import numpy as np
import pandas as pd
from enum import Enum


class Actions(Enum):
    REST = 0
    FLEX = 1
    EXT = 2
    RAD = 3
    ULN = 4
    SUP = 5
    PRO = 6


def seconds_to_idx(second, sampling_rate: int):
    """
    for labeling array, cyton board sampling rate
    """
    idx = second * sampling_rate
    return idx.astype(int)


def auto_label(label_file, data, bpm: int, sampling_rate: int, data_word_len: int, training_flag: bool):
    # 1. load the dataset using pandas reading a csv file
    dict_map = {'rest': Actions.REST,
                'right': Actions.FLEX,
                'left': Actions.EXT,
                'up':  Actions.RAD,
                'down': Actions.ULN,
                'out': Actions.SUP,
                'in': Actions.PRO,
                '-': Actions.REST}

    df = pd.read_csv(label_file)
    label = []  # Each label is 60/BPM second
    label_time = []
    temp_time = 0.0
    for i in range(len(df)):
        if df.loc[i]['Action'].lower() == 'rest':
            for j in range(df.loc[i]['Iterations']):
                label.append(Actions.REST.value)
                label_time.append(temp_time)
                temp_time += 60/bpm
        else:
            for j in range(df.loc[i]['Iterations']):
                label.append(dict_map[df.loc[i]['Action'].lower()].value)
                label_time.append(temp_time)
                temp_time += 60/bpm
                # if training_flag == False:  # Only random sequence test has rest in between
                #     label.append(Actions.REST.value)
                #     label_time.append(temp_time)
                #     temp_time += 60/bpm
    label = np.array(label)
    label_time = np.array(label_time)
    n_labels = data.shape[1]
    time = np.arange(0, n_labels) / sampling_rate
    total_time = n_labels / sampling_rate
    # All labels are 60/BPM second.
    # If total time is > len(label),
    # then we need to pad the labels with 0 in the beginning
    if total_time > (60/bpm)*len(label):
        label_time += (total_time - (60/bpm)*len(label))
        np.insert(label_time, 0, 0)
        label = np.concatenate(
            (np.zeros(int(total_time - (60/bpm)*len(label))), label))

    # Each label corresponds to a data_sentence. Divide data into data_sentence
    data_sentences = []
    label_sentences = []
    max_sentence_length = 0
    for i, start_time in enumerate(label_time):
        end_time = start_time + (60/bpm)
        start_idx = seconds_to_idx(start_time, sampling_rate)
        end_idx = seconds_to_idx(end_time, sampling_rate)
        data_sentence = data[:, start_idx:end_idx]
        # Divide data_sentence into data_word
        data_words = []
        label_words = []
        for j in range(0, data_sentence.shape[1], data_word_len):
            # Skip the last word if it is not long enough
            if data_sentence[:, j:j+data_word_len].shape[1] < data_word_len:
                break
            data_words.append(data_sentence[:, j: j+data_word_len])
            label_words.append(label[i])
            max_sentence_length = max(max_sentence_length, len(data_words))
        # data_words is a 3D array of shape (n_words, n_channels, data_word_len)

        if len(data_words) > 0:
            data_words = np.array(data_words)
            data_sentences.append(data_words)
            label_sentences.append(label_words)

    n_sentences = len(data_sentences)
    padded_data_sentences = np.zeros((
        n_sentences, max_sentence_length, 3, data_word_len))
    padded_label_sentences = -1 * np.ones((n_sentences, max_sentence_length))
    # Pad with -1 label to denote padding cut off
    for i in range(n_sentences):
        padded_data_sentences[i, : len(data_sentences[i])] = data_sentences[i]
        padded_label_sentences[i, : len(
            label_sentences[i])] = label_sentences[i]

    # If any input is nan, replace it with 0, and replace the corresponding label with -1
    is_nan = np.isnan(padded_data_sentences)
    # if any is_nan, print a statement
    if np.any(is_nan):
        print("There are nan in the data!")
    padded_data_sentences[is_nan] = 0
    padded_label_sentences[is_nan[:, :, 0, 0]] = -1
    return padded_data_sentences, padded_label_sentences
