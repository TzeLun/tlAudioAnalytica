import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import windows

# to automatically label the spectrogram based on the value of its force

# to filter the force values to get an estimate of their values within a short time frame

# to tally the sampling rate between audio and force sampling for coherent sampling in data preparation

# reconstruct audio from spectrogram (spectrogram inversion)

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


# Main management object for all audio handlers.
# The main object for automatic audio analysis and preprocessing for deep learning
class TLAudioHandlerManager:
    workspace = []  # to store multiple audio handlers

    # Provide the constructor with an iterables (list, np.array) of path to audio files
    def __init__(self, arr_path):
        for path in arr_path:
            self.workspace.append(TLAudioHandler(path))


# Main audio handler for audio data analysis and data preparation
class TLAudioHandler:
    path = ''  # path to folder for audio file
    fs = 44100  # sampling rate
    y = np.array([])  # time series audio data, initialize as np array
    y_seg = np.array([])  # segmented time series audio data
    S_seg = np.array([])  # to store the segmented audio's stft magnitude and phase
    S = np.array([])  # to store the main audio's stft magnitude and phase
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
                     'dpi': 96}
    stftProperties = {'window': 'hann',
                      'win_length': 256,
                      'n_fft': 512,
                      'hop_length': 128}
    specProperties = {'sr': 22050,
                      'n_mel': 128,
                      'power': 2.0,
                      'fmin': 0,
                      'fmax': 44100,
                      'db_ref': np.max}

    # Constructor to initialize the audio handler. One audio file per handler
    def __init__(self, path):
        self.path = path
        self.load()

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

    # Load the audio wave into a numpy ndarray time series data.
    def load(self, start=0.0, duration=None, fs=44100):
        self.y, self.fs = librosa.load(self.path, sr=fs, offset=start, duration=duration)

    # Resample the audio time series data using a target sampling rate
    def resample(self, target):
        self.y = librosa.resample(self.y, orig_sr=self.fs, target_sr=target)
        self.fs = target

    # Segment the audio into smaller pieces
    def segment_audio(self, start=0.0, duration=None):
        start_index = int(start * self.fs)  # point to begin sampling the sub-array
        if duration is None:
            end_index = int(self.get_duration() * self.fs)  # segment the audio till the end
            self.y_seg = self.y[start_index:end_index]
        else:
            if (start + duration) > self.get_duration():
                print("WARNING: DURATION EXCEEDS THE TOTAL TIME OF THE AUDIO")
                print("PROVIDED DURATION IS TRUNCATED TO THE MAXIMUM AUDIO TIME")
                end_index = int(self.get_duration() * self.fs)  # segment the audio till the end
                self.y_seg = self.y[start_index:end_index]
            else:
                end_index = int((start + duration) * self.fs)  # segment the audio till the end
                self.y_seg = self.y[start_index:end_index]

    # Overwrite the main audio tensor as the segmented one
    def set_segment_as_main(self):
        self.y = self.y_seg

    # Get the time series audio data with its sampling rate
    def get_audio_data(self):
        return self.y, self.fs

    # Get the time series audio segmented data with its sampling rate
    def get_audio_segment(self):
        return self.y_seg, self.fs

    # Get the duration of the audio signal or spectrogram
    def get_duration(self):
        return librosa.get_duration(y=self.y, sr=self.fs)

    # Get the duration of the audio signal or spectrogram
    def get_segment_duration(self):
        return librosa.get_duration(y=self.y_seg, sr=self.fs)

    # Perform short-time fourier transform on the audio signal, computes magnitude and phase.
    def stft(self, is_segment=False):
        if is_segment:
            S = librosa.stft(self.y_seg,
                             n_fft=self.stftProperties['n_fft'],
                             hop_length=self.stftProperties['hop_length'],
                             win_length=self.stftProperties['win_length'],
                             window=self.stftProperties['window'])
            self.S_seg = np.array([np.abs(S), np.angle(S)])
        else:
            S = librosa.stft(self.y,
                             n_fft=self.stftProperties['n_fft'],
                             hop_length=self.stftProperties['hop_length'],
                             win_length=self.stftProperties['win_length'],
                             window=self.stftProperties['window'])
            self.S = np.array([np.abs(S), np.angle(S)])

    # Retrieve magnitude of the signal's stft, able to select the main or segmented audio
    def get_stft_magnitude(self, is_segment=False):
        if is_segment:
            return self.S_seg[0]
        else:
            return self.S[0]

    # Retrieve phase of the signal's stft, able to select the main or segmeneted audio
    def get_stft_phase(self, is_segment=False):
        if is_segment:
            return self.S_seg[1]
        else:
            return self.S[1]

    # Plot the audio waveform in magnitude-time domain, requires matplotlib
    def plot_waveform(self, annotation=False, is_segment=False):
        y = self.y
        if is_segment:
            y = self.y_seg

        t = np.multiply(np.arange(0, len(y)), self.get_duration() / float(len(y)-1))
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
            plt.show()

    # Specifically save the plot or figure as an image
    # Filename does not need to have the format extension. Default format is PNG
    # The default image should only contain the plot without the axes, title and margins
    # To include the margins, set 'tight' to False
    # Default size of image is w x h : 256 x 256 (tuple format).
    # Default DPI is 96. Make sure to change it if it is different from the default
    def save_waveform_as_image(self, filename, file_format='.png',
                               size=figProperties['figsize'],
                               annotation=False, tight=True, is_segment=False):
        y = self.y
        if is_segment:
            y = self.y_seg

        t = np.multiply(np.arange(0, len(y)), self.get_duration() / float(len(y) - 1))
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
    def plot_mel_spectrogram(self, annotation=False, is_segment=False):
        if is_segment:
            if self.S_seg.size == 0:
                self.stft(is_segment)
                D = self.S_seg[0] ** self.specProperties['power']
            else:
                D = self.S_seg[0] ** self.specProperties['power']
        else:
            if self.S.size == 0:
                self.stft(is_segment)
                D = self.S[0] ** self.specProperties['power']
            else:
                D = self.S[0] ** self.specProperties['power']

        S = librosa.feature.melspectrogram(S=D,
                                           sr=self.specProperties['sr'],
                                           n_mels=self.specProperties['n_mel'])
        S_db = librosa.power_to_db(S, ref=self.specProperties['db_ref'])  # convert from power to decibel scale
        print(np.shape(S_db))
        if annotation:
            librosa.display.specshow(S_db,
                                     x_axis='time',
                                     y_axis='mel',
                                     sr=self.specProperties['sr'],
                                     fmin=self.specProperties['fmin'],
                                     fmax=self.specProperties['fmax'],
                                     hop_length=self.stftProperties['hop_length'])
            plt.title('Mel frequency spectrogram')
            plt.colorbar()
            plt.show()

        else:
            librosa.display.specshow(S_db,
                                     x_axis='time',
                                     y_axis='mel',
                                     sr=self.fs,
                                     fmin=self.specProperties['fmin'],
                                     fmax=self.specProperties['fmax'],
                                     hop_length=self.stftProperties['hop_length'])
            plt.show()

    # Save the mel_frequency spectrogram as an image
    def save_mel_spectrogram_as_image(self, filename, file_format='.png',
                                      size=figProperties['figsize'],
                                      annotation=False, tight=True, is_segment=False):
        if is_segment:
            if self.S_seg.size == 0:
                self.stft(is_segment)
                D = self.S_seg[0] ** self.specProperties['power']
            else:
                D = self.S_seg[0] ** self.specProperties['power']
        else:
            if self.S.size == 0:
                self.stft(is_segment)
                D = self.S[0] ** self.specProperties['power']
            else:
                D = self.S[0] ** self.specProperties['power']

        S = librosa.feature.melspectrogram(S=D,
                                           sr=self.specProperties['sr'],
                                           n_mels=self.specProperties['n_mel'])
        S_db = librosa.power_to_db(S, ref=self.specProperties['db_ref'])  # convert from power to decibel scale

        if annotation:
            plt.figure(figsize=(size[0] / self.figProperties['dpi'],
                                size[1] / self.figProperties['dpi']),
                       dpi=self.figProperties['dpi'])
            librosa.display.specshow(S_db,
                                     x_axis='time',
                                     y_axis='mel',
                                     sr=self.specProperties['sr'],
                                     fmin=self.specProperties['fmin'],
                                     fmax=self.specProperties['fmax'],
                                     hop_length=self.stftProperties['hop_length'])
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
            librosa.display.specshow(S_db,
                                     x_axis='time',
                                     y_axis='mel',
                                     sr=self.fs,
                                     fmin=self.specProperties['fmin'],
                                     fmax=self.specProperties['fmax'],
                                     hop_length=self.stftProperties['hop_length'])
            plt.axis('off')
            if tight:
                plt.tight_layout(pad=0)
                plt.savefig(filename + file_format,
                            dpi=self.figProperties['dpi'])
            else:
                plt.savefig(filename + file_format,
                            dpi=self.figProperties['dpi'])

