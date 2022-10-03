import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import windows

# to automatically label the spectrogram based on the value of its force

# to filter the force values to get an estimate of their values within a short time frame

# to tally the sampling rate between audio and force sampling for coherent sampling in data preparation

# reconstruct audio from spectrogram (spectrogram inversion)


# Load the audio wave into a numpy ndarray time series data.
def load(path, start=0.0, duration=None, sr=None):
    y, sr_ = librosa.load(path, sr=sr, offset=start, duration=duration)
    return y, sr_


# Normalize audio waveform volume or spectrogram power to a range between [-1 1]
def normalize(y):
    ref = np.max(np.abs(y))
    return np.divide(y, ref)


# Segment the audio into smaller pieces
def segment_audio(y, sr, start=0.0, duration=None):
    start_index = int(start * sr)  # point to begin sampling the sub-array
    if duration is None:
        end_index = int(get_duration(y, sr) * sr)  # segment the audio till the end
        return y[start_index:end_index]
    else:
        if (start + duration) > get_duration(y, sr):
            print("WARNING: DURATION EXCEEDS THE TOTAL TIME OF THE AUDIO")
            print("PROVIDED DURATION IS TRUNCATED TO THE MAXIMUM AUDIO TIME")
            end_index = int(get_duration(y, sr) * sr)  # segment the audio till the end
            return y[start_index:end_index]
        else:
            end_index = int((start + duration) * sr)  # segment the audio till the end
            return y[start_index:end_index]


# Convert a single mel spectrogram data to a suitable format for CNN use
def mel_spectrogram_to_cnn_data_format(S):
    return S[..., np.newaxis]


# Get the duration of the audio signal or spectrogram
def get_duration(y, sr):
    return librosa.get_duration(y=y, sr=sr)


# Provide the function with an iterable (list, np.array) of path to audio files
# Give a value for sr if the audio should be loaded differently from its native
# sampling rate.
# Warning: make sure all recorded audio are sampled at the same rate
# if using the native sampling rate, (sr = None), to avoid inconsistency.
def mul_load(arr_path, sr=None):
    Y = []
    SR = []
    sr_ = 44100
    for path in arr_path:
        y, sr_ = load(path, sr=sr)
        Y.append(y)
        SR.append(sr_)
    audio_handler = TLAudioHandler()
    audio_handler.set_sr(sr_)
    audio_handler.set_fmax(sr_ / 2)
    return np.array(Y), SR, audio_handler


# compute mel spectrogram given a list/array of waveform
def mul_compute_mel_spectrogram(Y, audio_handler):
    return np.array([audio_handler.mel_spectrogram(y) for y in Y])


# Normalize multiple mel spectrogram respectively
def mul_normalize_mel_spectrogram(S):
    return np.array([normalize(s) for s in S])


# Converts the given mel spectrogram into data format for use in CNN
# Requires computing mel spectrogram beforehand
def mul_mel_spectrogram_to_cnn_data_format(S):
    return np.array([s[..., np.newaxis] for s in S])


# Computes the mel spectrogram for each waveform first
# then converts them into data format for use in CNN
# Directly input the audio waveform array
def mul_mel_spectrogram_for_cnn(Y, audio_handler, normalization=False):
    return np.array([audio_handler.mel_spectrogram_for_cnn(y, normalization)[..., np.newaxis]
                     for y in Y])


# Plot a single or all the waveform in a single figure
# Able to select the number and also which plot to display by giving a list of indices
# To choose to plot an individual waveform, either input a list with 1 index or
# directly input the index in integer format.
# To plot all, set choice as 'all'
# To plot only the desired ones, choice is a list containing the desired indices
def disp_multiple_waveform(Y, SR, audio_handler, choice='all',
                           orientation='row-major', annotation=False):
    if type(choice) is int:
        audio_handler.plot_waveform(Y[choice],
                                    SR[choice],
                                    annotation, hold=True)
        plt.show()
    elif choice != 'all':
        if orientation == 'row-major':
            for ind in range(len(choice)):
                plt.subplot(len(choice), 1, ind + 1)
                audio_handler.plot_waveform(Y[choice[ind]],
                                            SR[choice[ind]],
                                            annotation, hold=True)
            plt.subplots_adjust(left=audio_handler.figProperties['subplot_padding'][0],
                                right=audio_handler.figProperties['subplot_padding'][1],
                                bottom=audio_handler.figProperties['subplot_padding'][2],
                                top=audio_handler.figProperties['subplot_padding'][3],
                                wspace=audio_handler.figProperties['subplot_padding'][4],
                                hspace=audio_handler.figProperties['subplot_padding'][5])
            plt.show()
        elif orientation == 'column-major':
            for ind in range(len(choice)):
                plt.subplot(1, len(choice), ind + 1)
                audio_handler.plot_waveform(Y[choice[ind]],
                                            SR[choice[ind]],
                                            annotation, hold=True)
            plt.subplots_adjust(left=audio_handler.figProperties['subplot_padding'][0],
                                right=audio_handler.figProperties['subplot_padding'][1],
                                bottom=audio_handler.figProperties['subplot_padding'][2],
                                top=audio_handler.figProperties['subplot_padding'][3],
                                wspace=audio_handler.figProperties['subplot_padding'][4],
                                hspace=audio_handler.figProperties['subplot_padding'][5])
            plt.show()
    elif choice == 'all':
        if orientation == 'row-major':
            for ind in range(len(Y)):
                plt.subplot(len(Y), 1, ind + 1)
                audio_handler.plot_waveform(Y[ind],
                                            SR[ind],
                                            annotation, hold=True)
            plt.subplots_adjust(left=audio_handler.figProperties['subplot_padding'][0],
                                right=audio_handler.figProperties['subplot_padding'][1],
                                bottom=audio_handler.figProperties['subplot_padding'][2],
                                top=audio_handler.figProperties['subplot_padding'][3],
                                wspace=audio_handler.figProperties['subplot_padding'][4],
                                hspace=audio_handler.figProperties['subplot_padding'][5])
            plt.show()
        elif orientation == 'column-major':
            for ind in range(len(Y)):
                plt.subplot(1, len(Y), ind + 1)
                audio_handler.plot_waveform(Y[ind],
                                            SR[ind],
                                            annotation, hold=True)
            plt.subplots_adjust(left=audio_handler.figProperties['subplot_padding'][0],
                                right=audio_handler.figProperties['subplot_padding'][1],
                                bottom=audio_handler.figProperties['subplot_padding'][2],
                                top=audio_handler.figProperties['subplot_padding'][3],
                                wspace=audio_handler.figProperties['subplot_padding'][4],
                                hspace=audio_handler.figProperties['subplot_padding'][5])
            plt.show()


# Plot a single or all the spectrogram in a single figure
# Able to select the number and also which plot to display by giving a list of indices
# To choose to plot an individual waveform, either input a list with 1 index or
# directly input the index in integer format.
# To plot all, set choice as 'all'
# To plot only the desired ones, choice is a list containing the desired indices
def disp_multiple_mel_spectrogram(S, audio_handler, choice='all',
                                  orientation='row-major', annotation=False):
    if type(choice) is int:
        audio_handler.plot_mel_spectrogram(S[choice], annotation, hold=True)
        plt.show()
    elif choice != 'all':
        if orientation == 'row-major':
            for ind in range(len(choice)):
                plt.subplot(len(choice), 1, ind + 1)
                audio_handler.plot_mel_spectrogram(S[choice[ind]], annotation, hold=True)
            plt.subplots_adjust(left=audio_handler.figProperties['subplot_padding'][0],
                                right=audio_handler.figProperties['subplot_padding'][1],
                                bottom=audio_handler.figProperties['subplot_padding'][2],
                                top=audio_handler.figProperties['subplot_padding'][3],
                                wspace=audio_handler.figProperties['subplot_padding'][4],
                                hspace=audio_handler.figProperties['subplot_padding'][5])
            plt.show()
        elif orientation == 'column-major':
            for ind in range(len(choice)):
                plt.subplot(1, len(choice), ind + 1)
                audio_handler.plot_mel_spectrogram(S[choice[ind]], annotation, hold=True)
            plt.subplots_adjust(left=audio_handler.figProperties['subplot_padding'][0],
                                right=audio_handler.figProperties['subplot_padding'][1],
                                bottom=audio_handler.figProperties['subplot_padding'][2],
                                top=audio_handler.figProperties['subplot_padding'][3],
                                wspace=audio_handler.figProperties['subplot_padding'][4],
                                hspace=audio_handler.figProperties['subplot_padding'][5])
            plt.show()
    elif choice == 'all':
        if orientation == 'row-major':
            for ind in range(len(S)):
                plt.subplot(len(S), 1, ind + 1)
                audio_handler.plot_mel_spectrogram(S[ind], annotation, hold=True)
            plt.subplots_adjust(left=audio_handler.figProperties['subplot_padding'][0],
                                right=audio_handler.figProperties['subplot_padding'][1],
                                bottom=audio_handler.figProperties['subplot_padding'][2],
                                top=audio_handler.figProperties['subplot_padding'][3],
                                wspace=audio_handler.figProperties['subplot_padding'][4],
                                hspace=audio_handler.figProperties['subplot_padding'][5])
            plt.show()
        elif orientation == 'column-major':
            for ind in range(len(S)):
                plt.subplot(1, len(S), ind + 1)
                audio_handler.plot_mel_spectrogram(S[ind], annotation, hold=True)
            plt.subplots_adjust(left=audio_handler.figProperties['subplot_padding'][0],
                                right=audio_handler.figProperties['subplot_padding'][1],
                                bottom=audio_handler.figProperties['subplot_padding'][2],
                                top=audio_handler.figProperties['subplot_padding'][3],
                                wspace=audio_handler.figProperties['subplot_padding'][4],
                                hspace=audio_handler.figProperties['subplot_padding'][5])
            plt.show()


# Save the mel spectrogram as an image:


# Main audio handler for audio data analysis and data preparation
class TLAudioHandler:
    figProperties = {'title': 'Audio',
                     'xlabel': 'Time (s)',
                     'ylabel': 'Magnitude',
                     'label': 'figure',
                     'color': 'teal',
                     'marker': None,
                     'linestyle': '-',
                     'linewidth': 1.0,
                     'markersize': 0.0,
                     'markerfacecolor': 'teal',
                     'xlim': (None, None),
                     'ylim': (None, None),
                     'figsize': (256, 256),
                     'dpi': 96,
                     'colorbar_range': (None, None),
                     'cmap': 'jet',
                     'subplot_padding': [None, None, None, None, None, 0.5]}
    stftProperties = {'window': 'hann',
                      'win_length': 512,
                      'n_fft': 1024,
                      'hop_length': 256}
    specProperties = {'sr': 44100,
                      'n_mel': 128,
                      'power': 2.0,
                      'fmin': 0,
                      'fmax': 22050,
                      'db_ref': 1.0}
    # every possible window functions for STFT based on scipy
    stft_window = {'barthann': windows.barthann,
                   'bartlett': windows.bartlett,
                   'blackman': windows.blackman,
                   'blackmanharris': windows.blackmanharris,
                   'bohman': windows.bohman,
                   'boxcar': windows.boxcar,
                   'chebwin': windows.chebwin,
                   'cosine': windows.cosine,
                   'exponential': windows.exponential,
                   'flattop': windows.flattop,
                   'gaussian': windows.gaussian,
                   'general_cosine': windows.general_cosine,
                   'general_gaussian': windows.general_gaussian,
                   'general_hamming': windows.general_hamming,
                   'hamming': windows.hamming,
                   'hann': windows.hann,
                   'kaiser': windows.kaiser,
                   'kaiser_bessel_derived': windows.kaiser_bessel_derived,
                   'nuttall': windows.nuttall,
                   'parzen': windows.parzen,
                   'taylor': windows.taylor,
                   'triang': windows.triang,
                   'tukey': windows.tukey}

    # Set title of the plot
    def set_title(self, title):
        self.figProperties['title'] = title

    # Set xlabel of the plot
    def set_xlabel(self, xlabel):
        self.figProperties['xlabel'] = xlabel

    # Set ylabel of the plot
    def set_ylabel(self, ylabel):
        self.figProperties['ylabel'] = ylabel

    # Set the graph label of the plot for legend
    def set_label(self, label):
        self.figProperties['label'] = label

    # Set color of the plot
    def set_color(self, color):
        self.figProperties['color'] = color

    # Set marker of the plot
    def set_marker(self, marker):
        self.figProperties['marker'] = marker

    # Set line style of the plot
    def set_linestyle(self, linestyle):
        self.figProperties['linestyle'] = linestyle

    # Set line width of the plot
    def set_linewidth(self, linewidth):
        self.figProperties['linewidth'] = linewidth

    # Set marker size of the plot
    def set_markersize(self, markersize):
        self.figProperties['markersize'] = markersize

    # Set the marker face color of the plot
    def set_markerfacecolor(self, markerfacecolor):
        self.figProperties['markerfacecolor'] = markerfacecolor

    # Set xlim of the plot. In general, this is a tuple
    def set_xlim(self, xlim):
        self.figProperties['xlim'] = xlim

    # Set ylim of the plot. In general, this is a tuple
    def set_ylim(self, ylim):
        self.figProperties['ylim'] = ylim

    # Set size of figure of the plot. This will affect the size of the image saved
    # Given as a tuple: (w, h)
    def set_figsize(self, figsize):
        self.figProperties['figsize'] = figsize

    # Set dpi of the plot. Important for figure or image sizing
    # To get your monitor dpi, try this link:
    # https://www.infobyip.com/detectmonitordpi.php
    def set_dpi(self, dpi):
        self.figProperties['dpi'] = dpi

    # Set the colorbar range, (vmin, vmax)
    def set_colorbar_range(self, range):
        self.figProperties['colorbar_range'] = range

    # Set color map, default: 'jet'
    def set_cmap(self, cm):
        self.figProperties['cmap'] = cm

    # Set padding of subplots, set the values by specifying the arguments, or
    # Give a list of at most 6 floating point values,
    # each pertaining to the order of the padding values as follows:
    # [left, right, bottom, top, wspace, hspace]
    # Any list size, n, smaller than 6 will just edit the first n padding values
    # To edit only one padding parameter, give the name of the padding in strings/char
    # for the option argument. Then set the padding value through pad argument
    def set_subplot_padding(self, option, pad=None):
        if type(option) is list:
            for i in range(len(option)):
                self.figProperties['subplot_padding'][i] = option[i]
        elif option == 'left':
            self.figProperties['subplot_padding'][0] = pad
        elif option == 'right':
            self.figProperties['subplot_padding'][1] = pad
        elif option == 'bottom':
            self.figProperties['subplot_padding'][2] = pad
        elif option == 'top':
            self.figProperties['subplot_padding'][3] = pad
        elif option == 'wspace':
            self.figProperties['subplot_padding'][4] = pad
        elif option == 'hspace':
            self.figProperties['subplot_padding'][5] = pad

    # Set the window type for STFT.
    # Available windows are according to scipy.signal.windows
    # Set it via inputting the name of the window in strings
    def set_window(self, window):
        self.stftProperties['window'] = window

    # Set the window length for STFT. Length is measured in samples
    # Give an integer (best at a power of 2).
    def set_winlength(self, winlength):
        self.stftProperties['win_length'] = winlength

    # Set the number of FFT, nfft. dType: integer (samples)
    # Larger nfft corresponds to longer computation time
    def set_nfft(self, nfft):
        self.stftProperties['n_fft'] = nfft

    # Set the hop length of a window function. dType: integer (samples)
    def set_hop_length(self, hop_length):
        self.stftProperties['hop_length'] = hop_length

    # Set the sampling rate for the spectrogram. Default at 44100
    def set_sr(self, sr):
        self.specProperties['sr'] = sr

    # Set the number of mel band filter for the spectrogram. Default at 128
    def set_nmel(self, n_mel):
        self.specProperties['n_mel'] = n_mel

    # set the power to raise the magnitude of the STFT.
    # dType: Float. Default: 2.0 (Power scale)
    def set_power(self, power):
        self.specProperties['power'] = power

    # set the minimum frequency to display the spectrogram plot. dType: Float > 0
    def set_fmin(self, fmin):
        self.specProperties['fmin'] = fmin

    # set the maximum frequency to display the spectrogram plot. dType: Float > 0
    def set_fmax(self, fmax):
        self.specProperties['fmax'] = fmax

    # Set the reference value for the power to dB conversion for spectrogram.
    # dType: float (magnitude of STFT). Numpy function also available, ie: np.max (default)
    def set_db_ref(self, ref):
        self.specProperties['db_ref'] = ref

    # Resample the audio time series data using a target sampling rate
    def resample(self, y, current_sr, target_sr):
        self.set_sr(target_sr)
        self.set_fmax(target_sr / 2)
        return librosa.resample(y, orig_sr=current_sr, target_sr=target_sr)

    # Perform short-time fourier transform on the audio signal, computes magnitude and phase.
    def stft(self, y):
        S = librosa.stft(y,
                         n_fft=self.stftProperties['n_fft'],
                         hop_length=self.stftProperties['hop_length'],
                         win_length=self.stftProperties['win_length'],
                         window=self.stftProperties['window'])
        return np.array([np.abs(S), np.angle(S)])

    # Compute the mel spectrogram from the audio data array. Computes stft within.
    def mel_spectrogram(self, y):
        D = self.stft(y)[0] ** self.specProperties['power']
        S = librosa.feature.melspectrogram(S=D,
                                           sr=self.specProperties['sr'],
                                           n_mels=self.specProperties['n_mel'])
        Sdb = librosa.power_to_db(S,
                                  ref=self.specProperties['db_ref'])  # convert from power to decibel scale
        # print(np.max(Sdb))
        return Sdb

    # get the mel spectrogram in the form that works with CNN: (H, W, Channels)
    def mel_spectrogram_for_cnn(self, y, normalization=False):
        if normalization:
            return normalize(self.mel_spectrogram(y))[..., np.newaxis]
        else:
            return self.mel_spectrogram(y)[..., np.newaxis]

    # Plot the audio waveform in magnitude-time domain, requires matplotlib
    def plot_waveform(self, y, sr=specProperties['sr'], annotation=False, hold=False):
        t = np.multiply(np.arange(0, len(y)), get_duration(y, sr) / float(len(y)-1))
        if annotation:
            plt.plot(t, y,
                     color=self.figProperties['color'],
                     linestyle=self.figProperties['linestyle'],
                     linewidth=self.figProperties['linewidth'],
                     marker=self.figProperties['marker'],
                     markersize=self.figProperties['markersize'],
                     markerfacecolor=self.figProperties['markerfacecolor'],
                     label=self.figProperties['label'])
            plt.title(self.figProperties['title'])
            plt.xlabel(self.figProperties['xlabel'])
            plt.ylabel(self.figProperties['ylabel'])
            plt.xlim(t[0], t[-1])
            if all(self.figProperties['ylim']):
                plt.ylim(self.figProperties['ylim'])
            if all(self.figProperties['xlim']):
                plt.xlim(self.figProperties['xlim'])
            if not hold:
                plt.show()
        else:
            plt.plot(t, y,
                     color=self.figProperties['color'],
                     linestyle=self.figProperties['linestyle'],
                     linewidth=self.figProperties['linewidth'],
                     marker=self.figProperties['marker'],
                     markersize=self.figProperties['markersize'],
                     markerfacecolor=self.figProperties['markerfacecolor'],
                     label=self.figProperties['label'])
            plt.xlim(t[0], t[-1])
            if all(self.figProperties['ylim']):
                plt.ylim(self.figProperties['ylim'])
            if all(self.figProperties['xlim']):
                plt.xlim(self.figProperties['xlim'])
            if not hold:
                plt.show()

    # Specifically save the plot or figure as an image
    # Filename does not need to have the format extension. Default format is PNG
    # The default image should only contain the plot without the axes, title and margins
    # To include the margins, set 'tight' to False
    # Default size of image is w x h : 256 x 256 (tuple format).
    # Default DPI is 96. Make sure to change it if it is different from the default
    def save_waveform_as_image(self, y, sr=specProperties['sr'],
                               filename='waveform', file_format='.png',
                               size=figProperties['figsize'],
                               annotation=False, tight=True):
        t = np.multiply(np.arange(0, len(y)), get_duration(y, sr) / float(len(y) - 1))
        if annotation:
            plt.figure(figsize=(size[0]/self.figProperties['dpi'],
                                size[1]/self.figProperties['dpi']),
                       dpi=self.figProperties['dpi'])
            plt.plot(t, y,
                     color=self.figProperties['color'],
                     linestyle=self.figProperties['linestyle'],
                     linewidth=self.figProperties['linewidth'],
                     marker=self.figProperties['marker'],
                     markersize=self.figProperties['markersize'],
                     markerfacecolor=self.figProperties['markerfacecolor'],
                     label=self.figProperties['label'])
            plt.title(self.figProperties['title'])
            plt.xlabel(self.figProperties['xlabel'])
            plt.ylabel(self.figProperties['ylabel'])
            plt.xlim(t[0], t[-1])
            if all(self.figProperties['ylim']):
                plt.ylim(self.figProperties['ylim'])
            if all(self.figProperties['xlim']):
                plt.xlim(self.figProperties['xlim'])
            if tight:
                plt.tight_layout(pad=0)
                plt.savefig(filename + file_format,
                            dpi=self.figProperties['dpi'])
            else:
                plt.savefig(filename + file_format,
                            dpi=self.figProperties['dpi'])
            plt.clf()

        else:
            plt.figure(figsize=(size[0] / self.figProperties['dpi'],
                                size[1] / self.figProperties['dpi']),
                       dpi=self.figProperties['dpi'])
            plt.plot(t, y,
                     color=self.figProperties['color'],
                     linestyle=self.figProperties['linestyle'],
                     linewidth=self.figProperties['linewidth'],
                     marker=self.figProperties['marker'],
                     markersize=self.figProperties['markersize'],
                     markerfacecolor=self.figProperties['markerfacecolor'],
                     label=self.figProperties['label'])
            plt.xlim(t[0], t[-1])
            plt.axis('off')
            if all(self.figProperties['ylim']):
                plt.ylim(self.figProperties['ylim'])
            if all(self.figProperties['xlim']):
                plt.xlim(self.figProperties['xlim'])
            if tight:
                plt.tight_layout(pad=0)
                plt.savefig(filename + file_format,
                            dpi=self.figProperties['dpi'])
            else:
                plt.savefig(filename + file_format,
                            dpi=self.figProperties['dpi'])
            plt.clf()

    # Plot the audio spectrum in mel_frequency-time domain, requires matplotlib
    # Requires computing the mel spectrogram beforehand.
    # Do not convert to the tensor form for CNN when trying to plot
    def plot_mel_spectrogram(self, S, annotation=False, hold=False):
        if annotation:
            librosa.display.specshow(S,
                                     x_axis='time',
                                     y_axis='mel',
                                     sr=self.specProperties['sr'],
                                     fmin=self.specProperties['fmin'],
                                     fmax=self.specProperties['fmax'],
                                     hop_length=self.stftProperties['hop_length'],
                                     vmin=self.figProperties['colorbar_range'][0],
                                     vmax=self.figProperties['colorbar_range'][1],
                                     cmap=self.figProperties['cmap'])
            plt.title('Mel frequency spectrogram')
            plt.colorbar()
            if not hold:
                plt.show()
        else:
            librosa.display.specshow(S,
                                     x_axis='time',
                                     y_axis='mel',
                                     sr=self.specProperties['sr'],
                                     fmin=self.specProperties['fmin'],
                                     fmax=self.specProperties['fmax'],
                                     hop_length=self.stftProperties['hop_length'],
                                     vmin=self.figProperties['colorbar_range'][0],
                                     vmax=self.figProperties['colorbar_range'][1],
                                     cmap=self.figProperties['cmap'])
            if not hold:
                plt.show()

    # Save the mel_frequency spectrogram as an image
    # Same as plot_mel_spectrogram but doesn't display anything
    # Arguments are similar to save_waveform_as_image()
    def save_mel_spectrogram_as_image(self, S, filename, file_format='.png',
                                      size=figProperties['figsize'],
                                      annotation=False, tight=True):

        if annotation:
            plt.figure(figsize=(size[0] / self.figProperties['dpi'],
                                size[1] / self.figProperties['dpi']),
                       dpi=self.figProperties['dpi'])
            librosa.display.specshow(S,
                                     x_axis='time',
                                     y_axis='mel',
                                     sr=self.specProperties['sr'],
                                     fmin=self.specProperties['fmin'],
                                     fmax=self.specProperties['fmax'],
                                     hop_length=self.stftProperties['hop_length'],
                                     vmin=self.figProperties['colorbar_range'][0],
                                     vmax=self.figProperties['colorbar_range'][1],
                                     cmap=self.figProperties['cmap'])
            plt.title('Mel frequency spectrogram')
            plt.colorbar()
            if tight:
                plt.tight_layout(pad=0)
                plt.savefig(filename + file_format,
                            dpi=self.figProperties['dpi'])
            else:
                plt.savefig(filename + file_format,
                            dpi=self.figProperties['dpi'])
        else:
            plt.figure(figsize=(size[0] / self.figProperties['dpi'],
                                size[1] / self.figProperties['dpi']),
                       dpi=self.figProperties['dpi'])
            librosa.display.specshow(S,
                                     x_axis='time',
                                     y_axis='mel',
                                     sr=self.specProperties['sr'],
                                     fmin=self.specProperties['fmin'],
                                     fmax=self.specProperties['fmax'],
                                     hop_length=self.stftProperties['hop_length'],
                                     vmin=self.figProperties['colorbar_range'][0],
                                     vmax=self.figProperties['colorbar_range'][1],
                                     cmap=self.figProperties['cmap'])
            plt.axis('off')
            if tight:
                plt.tight_layout(pad=0)
                plt.savefig(filename + file_format,
                            dpi=self.figProperties['dpi'])
            else:
                plt.savefig(filename + file_format,
                            dpi=self.figProperties['dpi'])


# ---------------------------- Data Augmentation function -------------------------------

# Input the spectrogram or mel spectrogram array, axis/unit format doesn't matter
# Up to user's liking. Format should NOT be a CNN image style, ie: (W, H, Channel)
# m_f is the number of frequency mask, usually set to 1 or 2
# F gives a random range of the mask coverage. The larger the F, potentially wider the mask
# WARNING: THIS FUNCTION WILL LEAD TO MUTATION OF THE INPUT SPECTROGRAM
def frequency_mask(S, m_f=2, F=2):
    # loop over to create m_f masks
    for i in range(m_f):
        # randomly select the number of frequency bands (width)
        f = int(np.random.uniform(0, F))
        # randomly select the starting mel band to apply the masking
        f0 = np.random.randint(0, S.shape[0] - f)
        # Mask the frequency band by setting the power values to 0 (could be anything else)
        S[f0:f0+f, :] = 0
    return S


# Input the spectrogram or mel spectrogram array, axis/unit format doesn't matter
# Up to user's liking. Format should NOT be a CNN image style, ie: (W, H, Channel)
# m_t is the number of time mask, usually set to 1 or 2
# T gives a random range of the mask coverage. The larger the T, potentially wider the mask
# WARNING: THIS FUNCTION WILL LEAD TO MUTATION OF THE INPUT SPECTROGRAM
def time_mask(S, m_t=2, T=2):
    # loop over to create m_t masks
    for i in range(m_t):
        # randomly select the number of frequency bands (width)
        t = int(np.random.uniform(0, T))
        # randomly select the starting time frame to apply the masking
        t0 = np.random.randint(0, S.shape[1] - t)
        # Mask the time band by setting the power values to 0 (could be anything else)
        S[:, t0:t0+t] = 0
    return S


# Time shifting for raw audio waveform, randomly shifts the audio waveform
# along the time domain.
def time_shift(y, sr, t=0.0):
    shift = int(t * sr)
    y0 = np.zeros(shift)
    if shift >= len(y):
        return y0
    else:
        y1 = y[0:(len(y) - shift)]
        return np.concatenate((y0, y1), axis=0)


# Amplitude scaling or gain scaling for raw audio waveform
def gain_scaling(y, gain=1.0):
    return np.multiply(y, gain)
