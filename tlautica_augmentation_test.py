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

#
# # -------------- Time shifting ----------------------------
# a_h.set_title('free running drill (time shifting)')
# y_shifted = tlautica.time_shift(y, sr, 1.0)
# # Waveform analysis
# Y = [y, y_shifted]
# SR = [sr, sr]
# tlautica.disp_multiple_waveform(Y, SR, a_h)
# # Mel Spectrogram analysis
# s_y_shifted = tlautica.normalize(a_h.mel_spectrogram(y_shifted))
# S = [s, s_y_shifted]
# tlautica.disp_multiple_mel_spectrogram(S, a_h)
#
#
# # ----------------- Time stretching --------------------------
# a_h.set_title('free running drill (time stretching, rate=2.0)')
# # Stretch up the audio by a factor of 2
# y_stretch = tlautica.time_stretch(y, 2)
# # Waveform analysis
# Y = [y, y_stretch]
# SR = [sr, sr]
# tlautica.disp_multiple_waveform(Y, SR, a_h)
# # Mel Spectrogram analysis
# s_y_stretch = tlautica.normalize(a_h.mel_spectrogram(y_stretch))
# S = [s, s_y_stretch]
# tlautica.disp_multiple_mel_spectrogram(S, a_h)
#
# a_h.set_title('free running drill (time stretching, rate=0.5)')
# # Compress the audio by a factor of 2
# y_stretch = tlautica.time_stretch(y, 0.5)
# # Waveform analysis
# Y = [y, y_stretch]
# SR = [sr, sr]
# tlautica.disp_multiple_waveform(Y, SR, a_h)
# # Mel Spectrogram analysis
# s_y_stretch = tlautica.normalize(a_h.mel_spectrogram(y_stretch))
# S = [s, s_y_stretch]
# tlautica.disp_multiple_mel_spectrogram(S, a_h)
#
#
# # ---------------- Gain Scaling -------------------------
# a_h.set_title('free running drill (gain scaling, k=2.0)')
# # Scale up the audio by a factor of 2
# y_scale = tlautica.gain_scaling(y, 2.0)
# # Waveform analysis
# Y = [y, y_scale]
# SR = [sr, sr]
# tlautica.disp_multiple_waveform(Y, SR, a_h)
# # Mel Spectrogram analysis
# s_y_scale = tlautica.normalize(a_h.mel_spectrogram(y_scale))
# S = [s, s_y_scale]
# tlautica.disp_multiple_mel_spectrogram(S, a_h)
#
# a_h.set_title('free running drill (gain scaling, k=0.5)')
# # Scale down the audio by a factor of 2
# y_scale = tlautica.gain_scaling(y, 0.5)
# # Waveform analysis
# Y = [y, y_scale]
# SR = [sr, sr]
# tlautica.disp_multiple_waveform(Y, SR, a_h)
# # Mel Spectrogram analysis
# s_y_scale = tlautica.normalize(a_h.mel_spectrogram(y_scale))
# S = [s, s_y_scale]
# tlautica.disp_multiple_mel_spectrogram(S, a_h)
#
#
# # ----------------- Add white noise -------------------------
# a_h.set_title('free running drill (white noise, SNR=2.0, k=1.0)')
# # Add white noise of with a SNR of 2.0, k = 1.0
# y_noisy = tlautica.add_white_noise(y, 2.0, 1.0)
# # Waveform analysis
# Y = [y, y_noisy]
# SR = [sr, sr]
# tlautica.disp_multiple_waveform(Y, SR, a_h)
# # Mel Spectrogram analysis
# s_y_noisy = tlautica.normalize(a_h.mel_spectrogram(y_noisy))
# S = [s, s_y_noisy]
# tlautica.disp_multiple_mel_spectrogram(S, a_h)
#
# a_h.set_title('free running drill (white noise, SNR=0.5, k=2.0)')
# # Add white noise of with a SNR of 0.5, k = 2.0
# y_noisy = tlautica.add_white_noise(y, 0.5, 2.0)
# # Waveform analysis
# Y = [y, y_noisy]
# SR = [sr, sr]
# tlautica.disp_multiple_waveform(Y, SR, a_h)
# # Mel Spectrogram analysis
# s_y_noisy = tlautica.normalize(a_h.mel_spectrogram(y_noisy))
# S = [s, s_y_noisy]
# tlautica.disp_multiple_mel_spectrogram(S, a_h)
#
#
# # -------------------- Pitch scaling ----------------------
# a_h.set_title('free running drill (Pitch scaling, nstep=2)')
# # Shift pitch up the audio by 2 half steps (semitones)
# y_scale = tlautica.pitch_scaling(y, sr, 2)
# # Waveform analysis
# Y = [y, y_scale]
# SR = [sr, sr]
# tlautica.disp_multiple_waveform(Y, SR, a_h)
# # Mel Spectrogram analysis
# s_y_scale = tlautica.normalize(a_h.mel_spectrogram(y_scale))
# S = [s, s_y_scale]
# tlautica.disp_multiple_mel_spectrogram(S, a_h)
#
# a_h.set_title('free running drill (Pitch scaling, nstep=6)')
# # Shift pitch up the audio by 6 half steps (semitones)
# y_scale = tlautica.pitch_scaling(y, sr, 6)
# # Waveform analysis
# Y = [y, y_scale]
# SR = [sr, sr]
# tlautica.disp_multiple_waveform(Y, SR, a_h)
# # Mel Spectrogram analysis
# s_y_scale = tlautica.normalize(a_h.mel_spectrogram(y_scale))
# S = [s, s_y_scale]
# tlautica.disp_multiple_mel_spectrogram(S, a_h)
#
#
# # ---------------- Frequency masking -------------------
# a_h.set_title('free running drill (frequency mask, m_f=2, F=27)')
# s_freq_mask = tlautica.frequency_mask(tlautica.normalize(a_h.mel_spectrogram(y)), 2, 27)
# S = [s, s_freq_mask]
# tlautica.disp_multiple_mel_spectrogram(S, a_h)
#
#
# # ----------------- Time masking -----------------------
# a_h.set_title('free running drill (time mask, m_t=2, T=27)')
# s_time_mask = tlautica.time_mask(tlautica.normalize(a_h.mel_spectrogram(y)), 2, 27)
# S = [s, s_time_mask]
# tlautica.disp_multiple_mel_spectrogram(S, a_h)


# ------------------ Add organic noise -----------------------------
path_to_noise = 'C:/Users/Tze Lun/data prelim/data_prelim/pressing_2.wav'
noise, sr_noise = tlautica.load(path_to_noise)
noise = tlautica.segment_audio(noise, sr_noise, 2.0, 2.0)
y_noisy = tlautica.add_noise(y, sr, (noise, sr_noise), t_in=4.0, k=2.0)
# tlautica.convert_to_audiofile('with_noise.wav', y_noisy, sr)
dt = tlautica.to_metadata(np.array([y_noisy]), sr, 'audio waveform array')
tlautica.save_json('polluted_signal', dt)
print(y.shape[0])
print(y_noisy.shape[0])
# waveform analysis
Y = [y, y_noisy]
SR = [sr, sr]
tlautica.disp_multiple_waveform(Y, SR, a_h)
# mel spectrogram analysis
s_y_noisy = tlautica.normalize(a_h.mel_spectrogram(y_noisy))
S = [s, s_y_noisy]
tlautica.disp_multiple_mel_spectrogram(S, a_h)

json_handler = tlautica.load_json('polluted_signal')
# waveform analysis
Y = [y, json_handler.data[0]]
SR = [sr, json_handler.sr]
tlautica.disp_multiple_waveform(Y, SR, a_h)

