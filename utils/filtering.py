import math
import numpy as np
import matplotlib.pyplot as plt
from brainflow.data_filter import DataFilter, DetrendOperations
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


def splice(raw, time_start, time_end, ch_start, ch_end):
    """
    Get raw data for the given time frame and channels

    :param raw: raw data from csv
    :param time_start: in seconds
    :param time_end: in seconds
    :param ch_start: integer
    :param ch_end: integer
    :return: selected raw data, channels as rows
    """
    # board set up
    board_id = BoardIds.CYTON_BOARD.value
    board = BoardShim(board_id, BrainFlowInputParams())

    # splice
    cols_per_sec = board.get_sampling_rate(board_id)
    # print(cols_per_sec)
    start_col = math.floor(time_start * cols_per_sec)
    end_col = math.floor(time_end * cols_per_sec)
    return raw[ch_start:(ch_end + 1), start_col:end_col]


def moving_average(y, w):
    """
    :param y: data to filter
    :param w: number of data points in window size
    :return: moving average array matching length of y
    """
    return np.convolve(y, np.ones(w), 'same') / w


def msv(filename, start, end, n_channels, str_date_config, str_dof, save_fig=False, plot=False):
    """
    Detrend, square, moving average, plot and return data from BCI channel 1 to n_channels

    :param filename: string name of BCI data 'myfile.csv'
    :param start: timestamp in seconds of start of recording snippet for 1 DOF
    :param end: timestamp in seconds of end of recording snippet for 1 DOF
    :param n_channels: number of channels to grab, assumes start at ch1
    :param str_date_config: string describing EMG configuration
    :param str_dof: string describing DOF
    :param save_fig: bool if generated figures should be saved
    :return all_msv: mean square value of selected data
    """
    # load data, iters as cols
    raw = DataFilter.read_file(filename)

    # get emg channels 1-n_channels data
    dof1 = splice(raw, start, end, 1, n_channels)
    # print(f"shape before detrend {dof1.shape}")

    all_msv = None
    if plot:
        fig2, ax2 = plt.subplots()

    for i in range(n_channels):
        # remove DC offset. detrend options LINEAR or CONSTANT
        DataFilter.detrend(dof1[i], DetrendOperations.LINEAR.value)
        # comprehension list
        dof1_sub = np.array([dof1[i, j]
                            for j in range(0, np.shape(dof1)[1], 1)])
        y = dof1_sub
        # plot original
        if plot:
            fig, (ax0, ax1) = plt.subplots(nrows=2)
            x = np.linspace(start, end, np.shape(y)[0])
            ax0.plot(x, y, color='red', label='signal')
            ax0.legend(loc='right')

        # square
        y_squared = np.square(y)

        # moving window average
        w = 20

        y_msv = moving_average(y_squared, w)
        x_msv = np.linspace(start, end, np.shape(
            y_msv)[0])  # close, lose end points
        # print(f"shape of x_msv {x_msv.shape}")

        # plot msv
        if plot:
            ax1.plot(x_msv, y_msv, 'blue', label='msv')
            ax1.legend(loc='right')
            # ax1.set_yscale("log")
            # ax1.set_ylim(-10000, 250000)

            # plot labels
            fig.suptitle(str_date_config + ' MSV ' + str_dof +
                         ' CH' + str(i + 1) + ' Window=' + str(w))
            fig.supxlabel('Session time (s)')
            fig.supylabel('Actuation (microV)')
            if save_fig:
                fig.savefig(str_date_config.lower() + '_msv_' +
                            str_dof.lower() + '_ch' + str(i + 1) + '_w' + str(w))

            # plot all stacked channels
            ax2.set_title(str_date_config + ' MSV ' +
                          str_dof + ' All Channels')
            ax2.plot(x_msv, y_msv, label='Ch' + str(i + 1))
            ax2.set_xlabel('Session time (s)')
            ax2.set_ylabel('Actuation (microV)')
            ax2.legend()
        # ax2.set_ylim(-10000, 300000)
        # ax2.set_yscale("log")
            if save_fig and i == n_channels - 1:
                print('gonna save stacked')
                fig2.savefig(str_date_config.lower() + '_msv_' +
                             str_dof.lower() + '_allch')

        # save to master array
        if type(all_msv) == type(None):
            all_msv = np.zeros((n_channels, np.shape(dof1_sub)[0]))
        all_msv[i, :np.shape(y_msv)[0]] = np.transpose(y_msv)

    return all_msv
