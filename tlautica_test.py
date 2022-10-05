import tlautica

path = 'C:/Users/Tze Lun/data prelim/data_prelim/free_running_drill_2.wav'
y, sr = tlautica.load(path)
# Initialize the audio handler
a_h = tlautica.TLAudioHandler()
# Able to configure the properties of the audio handler
a_h.set_sr(sr)  # setting the sr and fmax initially is mandatory
a_h.set_fmax(sr / 2)  # unless the default is already aligned with the data's
a_h.set_title('free running drill')
# Segment audio with a tlautica function
ys = tlautica.segment_audio(y, sr, 4.5, 0.1)
print(ys.shape)
ys_shifted = tlautica.time_shift(ys, sr, 0.0)
# ys_scaled = tlautica.gain_scaling(ys, 5)
print(ys_shifted.shape)
a_h.plot_waveform(ys, sr)
a_h.plot_waveform(ys_shifted, sr)
# One of the other function of the audio handler
# ys_new = a_h.resample(ys, sr, 22050)
# Strength of the audio handler. Separates syntax from the analytical work
# sdb = tlautica.normalize(a_h.mel_spectrogram(ys))
# sdb_f = tlautica.normalize(a_h.mel_spectrogram(ys))
# sdb_t = tlautica.normalize(a_h.mel_spectrogram(ys))
# sdb_time_mask = tlautica.time_mask(sdb_t, 2, 5)
# sdb_freq_mask = tlautica.frequency_mask(sdb_f, 2, 27)
# a_h.set_colorbar_range((-1.0, 0.05))
# tlautica.disp_multiple_mel_spectrogram([sdb, sdb_time_mask, sdb_freq_mask], a_h, annotation=True)
# a_h.save_waveform_as_image(ys, filename='hello3', size=(1024, 1024), annotation=True, tight=False)
# a_h.save_mel_spectrogram_as_image(sdb, 'hello5', size=(256, 256), annotation=False, tight=True)
# print(a_h.mel_spectrogram_for_cnn(ys, True).shape)

# arr_path = [
#     'C:/Users/Tze Lun/data prelim/data_prelim/ambient.wav',
#     'C:/Users/Tze Lun/data prelim/data_prelim/free_running_drill.wav',
#     'C:/Users/Tze Lun/data prelim/data_prelim/free_running_drill_2.wav',
#     'C:/Users/Tze Lun/data prelim/data_prelim/pressing_1.wav',
#     'C:/Users/Tze Lun/data prelim/data_prelim/pressing_2.wav',
#     'C:/Users/Tze Lun/data prelim/data_prelim/pressing_3.wav'
# ]
#
# an = {
#     'ambient': 0,
#     'free drilling soft': 1,
#     'free drilling': 2,
#     'drilling 1': 3,
#     'drilling 2': 4,
#     'drilling 3': 5
# }


