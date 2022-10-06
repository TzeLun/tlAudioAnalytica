import tlautica
import numpy as np

# ------------Load the audio file-------------------------------
path = 'C:/Users/Tze Lun/data prelim/data_prelim/free_running_drill_2.wav'
y, sr = tlautica.load(path)

# ------------Initialize the audio handler------------------
a_h = tlautica.TLAudioHandler()

# ------------Mandatory configuration--------------------------
a_h.set_sr(sr)  # setting the sr and fmax initially is mandatory
a_h.set_fmax(sr / 2)  # unless the default is already aligned with the data's
# Pre-computation of the mel spectrogram
s = tlautica.normalize(a_h.mel_spectrogram(y))
# Set color bar range to that between -1 to 1
a_h.set_colorbar_range((-1.0, 0.1))
a_h.set_ylim((-0.1, 0.1))

# -------------- Time shifting ----------------------------
# y_shifted = tlautica.time_shift(y, sr, 5.0)
# # Waveform analysis
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_waveform(y, sr, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (time shift = 5.0s)')
# a_h.plot_waveform(y_shifted, sr, annotation=True, hold=True)
# a_h.show()
# # Mel Spectrogram analysis
# s_y_shifted = tlautica.normalize(a_h.mel_spectrogram(y_shifted))
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_mel_spectrogram(s, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (time shift = 5.0s)')
# a_h.plot_mel_spectrogram(s_y_shifted, annotation=True, hold=True)
# a_h.show()


# # ----------------- Time stretching --------------------------
# # Stretch up the audio by a factor of 4
# y_stretch = tlautica.time_stretch(y, 4)
# print(y.shape[0])
# print(y_stretch.shape[0])
# # Waveform analysis
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_waveform(y, sr, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (time stretch, rate = 4.0)')
# a_h.plot_waveform(y_stretch, sr, annotation=True, hold=True)
# a_h.show()
# # Mel Spectrogram analysis
# s_y_stretch = tlautica.normalize(a_h.mel_spectrogram(y_stretch))
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_mel_spectrogram(s, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (time stretch, rate = 4.0)')
# a_h.plot_mel_spectrogram(s_y_stretch, annotation=True, hold=True)
# a_h.show()
#
# # Compress down the audio by a factor of 0.2
# y_stretch = tlautica.time_stretch(y, 0.2)
# print(y.shape[0])
# print(y_stretch.shape[0])
# # Waveform analysis
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_waveform(y, sr, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (time stretch, rate = 0.2)')
# a_h.plot_waveform(y_stretch, sr, annotation=True, hold=True)
# a_h.show()
# # Mel Spectrogram analysis
# s_y_stretch = tlautica.normalize(a_h.mel_spectrogram(y_stretch))
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_mel_spectrogram(s, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (time stretch, rate = 0.2)')
# a_h.plot_mel_spectrogram(s_y_stretch, annotation=True, hold=True)
# a_h.show()


# # ---------------- Gain Scaling -------------------------
# # Scale up the audio by a factor of 4
# y_aug = tlautica.gain_scaling(y, 4.0)
# # Waveform analysis
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_waveform(y, sr, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (gain scaling, k = 4.0)')
# a_h.plot_waveform(y_aug, sr, annotation=True, hold=True)
# a_h.show()
# # Mel Spectrogram analysis
# s_aug = tlautica.normalize(a_h.mel_spectrogram(y_aug))
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_mel_spectrogram(s, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (gain scaling, k = 4.0)')
# a_h.plot_mel_spectrogram(s_aug, annotation=True, hold=True)
# a_h.show()
#
# # Scale up the audio by a factor of 0.2
# y_aug = tlautica.gain_scaling(y, 0.2)
# # Waveform analysis
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_waveform(y, sr, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (gain scaling, k = 0.2)')
# a_h.plot_waveform(y_aug, sr, annotation=True, hold=True)
# a_h.show()
# # Mel Spectrogram analysis
# s_aug = tlautica.normalize(a_h.mel_spectrogram(y_aug))
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_mel_spectrogram(s, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (gain scaling, k = 0.2)')
# a_h.plot_mel_spectrogram(s_aug, annotation=True, hold=True)
# a_h.show()
#
#
# # ----------------- Add white noise -------------------------
# # Add white noise of with a SNR of 2.0, k = 1.0
# y_aug = tlautica.add_white_noise(y, 2.0, 1.0)
# # Waveform analysis
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_waveform(y, sr, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (white noise, SNR = 2.0, k = 1.0)')
# a_h.plot_waveform(y_aug, sr, annotation=True, hold=True)
# a_h.show()
# # Mel Spectrogram analysis
# s_aug = tlautica.normalize(a_h.mel_spectrogram(y_aug))
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_mel_spectrogram(s, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (white noise, SNR = 2.0, k = 1.0)')
# a_h.plot_mel_spectrogram(s_aug, annotation=True, hold=True)
# a_h.show()
# #
# # Add white noise of with SNR of 0.5, k = 4.0
# y_aug = tlautica.add_white_noise(y, 0.5, 4.0)
# # Waveform analysis
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_waveform(y, sr, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (white noise, SNR = 0.5, k = 4.0)')
# a_h.plot_waveform(y_aug, sr, annotation=True, hold=True)
# a_h.show()
# # Mel Spectrogram analysis
# s_aug = tlautica.normalize(a_h.mel_spectrogram(y_aug))
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_mel_spectrogram(s, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (white noise, SNR = 0.5, k = 4.0)')
# a_h.plot_mel_spectrogram(s_aug, annotation=True, hold=True)
# a_h.show()
#
#
# # -------------------- Pitch scaling ----------------------
# # Shift pitch up the audio by 2 half steps (semitones)
# y_aug = tlautica.pitch_scaling(y, sr, 2)
# print(y.shape[0])
# print(y_aug.shape[0])
# # Waveform analysis
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_waveform(y, sr, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (pitch scaling, nstep = 2)')
# a_h.plot_waveform(y_aug, sr, annotation=True, hold=True)
# a_h.show()
# # Mel Spectrogram analysis
# s_aug = tlautica.normalize(a_h.mel_spectrogram(y_aug))
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_mel_spectrogram(s, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (pitch scaling, nstep = 2)')
# a_h.plot_mel_spectrogram(s_aug, annotation=True, hold=True)
# a_h.show()
#
# # Shift pitch up the audio by 6 half steps (semitones)
# y_aug = tlautica.pitch_scaling(y, sr, 6)
# print(y.shape[0])
# print(y_aug.shape[0])
# # Waveform analysis
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_waveform(y, sr, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (pitch scaling, nstep = 6)')
# a_h.plot_waveform(y_aug, sr, annotation=True, hold=True)
# a_h.show()
# # Mel Spectrogram analysis
# s_aug = tlautica.normalize(a_h.mel_spectrogram(y_aug))
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_mel_spectrogram(s, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (pitch scaling, nstep = 6)')
# a_h.plot_mel_spectrogram(s_aug, annotation=True, hold=True)
# a_h.show()


# # ---------------- Frequency masking -------------------
# s_aug = tlautica.frequency_mask(tlautica.normalize(a_h.mel_spectrogram(y)), 2, 27)
# # Mel Spectrogram analysis
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_mel_spectrogram(s, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (frequency mask, m_f=2, F=27)')
# a_h.plot_mel_spectrogram(s_aug, annotation=True, hold=True)
# a_h.show()
#
#
# # ----------------- Time masking -----------------------
# s_aug_ = tlautica.time_mask(tlautica.normalize(a_h.mel_spectrogram(y)), 2, 27)
# # Mel Spectrogram analysis
# a_h.subplot(2, 1, 1)
# a_h.set_title('free running drill (original)')
# a_h.plot_mel_spectrogram(s, annotation=True, hold=True)
# a_h.subplot(2, 1, 2)
# a_h.set_title('free running drill (time mask, m_t=2, T=27)')
# a_h.plot_mel_spectrogram(s_aug_, annotation=True, hold=True)
# a_h.show()


# ------------------ Add organic noise -----------------------------
# Hello noise amplified by 3x
path_to_noise = 'hello.wav'
noise, sr_noise = tlautica.load(path_to_noise)
noise = tlautica.segment_audio(noise, sr_noise, 1.0)  # remove unwanted spike at the start
a_h.set_title('hello.wav')
a_h.plot_waveform(noise, sr_noise, annotation=True)
y_aug = tlautica.add_noise(y, sr, (noise, sr_noise), t_in=4.0, k=0.0)
tlautica.convert_to_audiofile('free_running_drill_with_hello.wav', y_aug, sr)
# Waveform analysis
a_h.subplot(2, 1, 1)
a_h.set_title('free running drill (original)')
a_h.plot_waveform(y, sr, annotation=True, hold=True)
a_h.subplot(2, 1, 2)
a_h.set_title('free running drill (with "hello.wav", t_in = 4s, k = 0.5)')
a_h.plot_waveform(y_aug, sr, annotation=True, hold=True)
a_h.show()
# mel spectrogram analysis
s_aug = tlautica.normalize(a_h.mel_spectrogram(y_aug))
a_h.subplot(2, 1, 1)
a_h.set_title('free running drill (original)')
a_h.plot_mel_spectrogram(s, annotation=True, hold=True)
a_h.subplot(2, 1, 2)
a_h.set_title('free running drill (with "hello.wav", t_in = 4s, k = 0.5)')
a_h.plot_mel_spectrogram(s_aug, annotation=True, hold=True)
a_h.show()

# ------------------------ Saving and loading dataset ---------------------------------
# dt = tlautica.to_metadata(np.array([data]), sr, 'audio waveform array')
# tlautica.save_json('polluted_signal', dt)
# json_handler = tlautica.load_json('polluted_signal')
# waveform analysis
# Y = [y, json_handler.data[0]]
# SR = [sr, json_handler.sr]
# tlautica.disp_multiple_waveform(Y, SR, a_h)

