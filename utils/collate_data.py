#!/usr/bin/env python3.8
'''
Written by: Saksham Consul 05/08/2023
Script to create dataset from the raw data
from datetime import datetime
'''
import os
import pandas as pd
import argparse
from datetime import datetime
from tqdm import tqdm
from utils.filtering import msv
from utils.specific_tools import auto_label
from utils.plotting import plot_graph


def main(args):
    # Set file paths
    raw_data_path = 'data/Raw/'
    labels_path = 'data/Labels/labels legend.csv'
    training_data_path = 'data/Processed/Training/processed_data.csv'

    # Load labels data
    labels_data = pd.read_csv(labels_path)

    # Create a dictionary of raw data file paths
    raw_data_files = {}
    for filename in os.listdir(raw_data_path):
        if filename.endswith('.csv'):
            raw_data_files[filename.split('.')[0]] = os.path.join(
                raw_data_path, filename)

    # Create empty dataframes for training and random data
    # Data frame has columns for sentence, label
    training_data = pd.DataFrame(columns=['sentence', 'label'])

    # Loop through each row in the labels data and extract corresponding raw data
    for index, row in tqdm(labels_data.iterrows(), total=labels_data.shape[0]):
        # Get the file name and label

        filename = row['CSV'].split('.')[0]
        label = 'data/Labels/' + row['Label csv file']
        input_date_str = filename.split('_')[1]
        input_date = datetime.strptime(input_date_str, '%Y-%m-%d')
        data_tag = input_date.strftime('%b%d')+'_'+str(row['BPM'])+'BPM'
        training_flag = True if row['Sequence'] == 'Training' else False
        # Check if raw data file exists for the file name
        if filename in raw_data_files:
            # Load raw data
            raw_data = msv(raw_data_files[filename], row['Time start'],
                           row['Time end'], 3, data_tag, 'All')
            # data_senteneces: (n_sentences, n_words, n_channels, data_word_len)
            # label_sentences: (n_sentences, n_words)
            data_sentences, label_sentences = auto_label(label, raw_data,
                                                         row['BPM'], args.sampling_rate, args.data_word_len, training_flag)
            # Plot the graph and save it
            plot_graph(data_sentences, label_sentences,
                       'data/Processed/'+filename+'.png')

            # Append data_sentences and corresponding labels to training data
            for s_idx, sentence in enumerate(data_sentences):
                training_data.loc[len(training_data)] = [
                    sentence, label_sentences[s_idx]]

    # Save processed training data
    training_folder = '/'.join(training_data_path.split('/')[:-1])
    if not os.path.exists(training_folder):
        os.makedirs(training_folder)
    training_data.to_csv(training_data_path)
    training_data.to_pickle(training_data_path.split('.')[0]+'.pkl')
    print("Number of sentence in Training dataset: ", training_data.shape[0])

    print("Collated complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_rate', type=int, default=250,
                        help='Sampling rate of the data')
    parser.add_argument('--data_word_len', type=int, default=25,
                        help='Length of the data word')
    args = parser.parse_args()
    main(args)
