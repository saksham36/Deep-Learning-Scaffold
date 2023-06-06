#!/usr/bin/env python3.8
'''
Written by: Saksham Consul 05/04/2023
Scripts needed for plotting
'''
import numpy as np
import matplotlib.pyplot as plt


def simple_plot(data, title, xlabel, ylabel, savepath, sampling_rate=250):
    '''Simple plot function'''
    plt.close('all')
    plt.figure()
    fig, ax = plt.subplots(1, 1)
    x = (1/sampling_rate)*np.arange(data.shape[1])
    ax.plot(x, data[0], x, data[1], x, data[2])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(['Channel 1', 'Channel 2', 'Channel 3'])
    plt.savefig(savepath)
    return


def plot_labels(labels, data, title, xlabel, ylabel, savepath, sampling_rate=250):
    '''
    Plot labels on top of the data
    '''
    plt.close('all')
    plt.figure()
    color_map = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'k', 5: 'm', 6: 'c'}
    fig, ax = plt.subplots(1, 1)
    x = (1/sampling_rate)*np.arange(data.shape[1])
    for i in range(labels.shape[0]):
        ax.axvline(i, 0, 5000, color=color_map[labels[i]], alpha=0.1)

    ax.plot(x, data[0], x, data[1], x, data[2])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(['Channel 1', 'Channel 2', 'Channel 3'])
    plt.savefig(savepath)
    return


def plot_graph(data_sentences, label_sentences, filename):
    n_sentences = data_sentences.shape[0]
    n_words = data_sentences.shape[1]
    n_channels = data_sentences.shape[2]
    data_word_len = data_sentences.shape[3]

    # Calculate the total length of the plot
    total_length = n_sentences * n_words * data_word_len

    # Get unique labels
    legend_name_map = {
        0: 'Rest',
        1: 'Flexion',
        2: 'Extension',
        3: 'Radial Deviation',
        4: 'Ulnar Deviation',
        5: 'Supination',
        6: 'Pronation',
        - 1: 'Pad'
    }
    unique_labels = [key for key in legend_name_map.keys()]
    unique_labels_name = [legend_name_map[key]
                          for key in legend_name_map.keys()]

    # Assign colors to labels
    label_colors = plt.cm.get_cmap('plasma', len(unique_labels))

    # Set up plot parameters
    fig, axs = plt.subplots(n_channels, 1, figsize=(10, n_channels * 5))

    for channel in range(n_channels):
        ax = axs[channel]

        # Iterate over sentences
        for sentence_idx in range(n_sentences):

            data_sentence = data_sentences[sentence_idx, :, channel, :]
            label_sentence = label_sentences[sentence_idx]

            # Plot label sentence
            for word_idx in range(n_words):
                word_data = data_sentence[word_idx]
                label = label_sentence[word_idx]
                color = label_colors(list(unique_labels).index(label))
                x = np.arange(len(word_data)) + word_idx * \
                    data_word_len + sentence_idx * n_words * data_word_len
                ax.axvspan(x[0], x[-1],
                           alpha=0.3, color=color)

            # Plot data sentence
            for word_idx in range(n_words):
                word_data = data_sentence[word_idx]
                x = np.arange(len(word_data)) + word_idx * \
                    data_word_len + sentence_idx * n_words * data_word_len
                ax.plot(x, np.log(word_data), 'b', linewidth=1)

                # Plot word boundary
                # word_boundary = (word_idx + 1) * data_word_len + \
                #     sentence_idx * n_words * data_word_len - 0.5
                # ax.axvline(x=word_boundary, color='r', linewidth=0.5)

            # Plot sentence boundary
            sentence_boundary = (sentence_idx + 1) * \
                n_words * data_word_len - 0.5
            ax.axvline(x=sentence_boundary, color='k', linewidth=2)

        # Set subplot title
        ax.set_title(f'Channel {channel+1}')

        # Set y-axis label
        ax.set_ylabel('Data')

        # Set x-axis limits
        ax.set_xlim(0, total_length+1)

        # Set y-axis limits
        ax.set_ylim(0,  # min(np.log(data_sentences[:, :, channel, :].flatten()))-1,
                    max(np.log(data_sentences[:, :, channel, :].flatten())) + 1)

        # Set x-axis tick labels
        # ax.set_xticks(np.arange(n_words) * data_word_len + data_word_len / 2)
        # ax.set_xticklabels(np.arange(1, n_words + 1))

        # Add label legends
        legend_patches = []
        for label in unique_labels:
            color = label_colors(list(unique_labels).index(label))
            patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3)
            legend_patches.append(patch)

        ax.legend(legend_patches, unique_labels_name)

    # Set shared x-axis label
    fig.text(0.5, 0.04, 'Word Index', ha='center')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the plot
    plt.savefig(filename)
