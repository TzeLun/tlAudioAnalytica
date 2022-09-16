import librosa
import matplotlib.pyplot as plt
import numpy as np

# to load the audio wave file (checked)

# to plot the audio wave as a magnitude vs time graph (checked)

# to plot and view spectrogram with/without margin, axes, colour bar (half checked) **

# to slice and sample a short frame of an audio signal across a user-defined range (checked)

# to save the spectrogram or waveform plot as image (half checked) **

# to automatically label the spectrogram based on the value of its force

# to filter the force values to get an estimate of their values within a short time frame

# to tally the sampling rate between audio and force sampling for coherent sampling in data preparation

# reconstruct audio from spectrogram (spectrogram inversion)


class TLAudioHandler:
    path = ''  # path to folder for audio file
    fs = 44100  # sampling rate
    y = np.array([])  # time series audio data, initialize as np array
    y_seg = np.array([])  # segmented time series audio data
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
                     'ylim': (None, None)}

    # Constructor to initialize the audio handler. One audio file per handler
    def __init__(self, path):
        self.path = path
        self.load()

    # Load the audio wave into a numpy ndarray time series data.
    def load(self, start=0.0, duration=None, fs=44100):
        self.y, self.fs = librosa.load(self.path, sr=fs, offset=start, duration=duration)

    # Resample the audio time series data using a target sampling rate
    def resample(self, target):
        self.y = librosa.resample(self.y, orig_sr=self.fs, target_sr=target)
        self.fs = target

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

    # Plot the audio waveform in magnitude-time domain, requires matplotlib
    def plot_waveform(self, annotation=False):
        t = np.multiply(np.arange(0, len(self.y)), self.get_duration() / float(len(self.y)-1))
        if annotation:
            plt.figure()
            plt.plot(t, self.y,
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
            plt.figure()
            plt.plot(t, self.y,
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
    def save_plot_as_image(self, filename, file_format='.png', annotation=False, tight=True):
        t = np.multiply(np.arange(0, len(self.y)), self.get_duration() / float(len(self.y) - 1))
        if annotation:
            plt.figure()
            plt.plot(t, self.y,
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
                plt.savefig(filename + file_format, bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig(filename + file_format)
            plt.clf()

        else:
            plt.figure()
            plt.plot(t, self.y,
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
            if tight:
                plt.axis('off')
                plt.savefig(filename + file_format, bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig(filename + file_format)
            plt.clf()



