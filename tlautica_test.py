import tlautica

path = 'C:/Users/Tze Lun/data prelim/data_prelim/free_running_drill_2.wav'
y, sr = tlautica.load(path)
# Initialize the audio handler
frdr = tlautica.TLAudioHandler()
# Able to configure the properties of the audio handler
frdr.set_sr(sr)  # setting the sr and fmax initially is mandatory
frdr.set_fmax(sr / 2)  # unless the default is already aligned with the data's
frdr.set_title('free running drill')
# Segment audio with a tlautica function
ys = tlautica.segment_audio(y, sr, 4.5, 0.1)
# One of the other function of the audio handler
ys_new = frdr.resample(ys, sr, 22050)
# Strength of the audio handler. Separates syntax from the analytical work
frdr.plot_waveform(ys, annotation=False)
sdb = frdr.mel_spectrogram(ys)
frdr.plot_mel_spectrogram(sdb, True)
frdr.save_waveform_as_image(ys, filename='hello3', size=(1024, 1024), annotation=True, tight=False)
frdr.save_mel_spectrogram_as_image(sdb, 'hello5', size=(256, 256), annotation=False, tight=True)
print(frdr.mel_spectrogram_for_cnn(ys, True).shape)

arr_path = [
    'C:/Users/Tze Lun/data prelim/data_prelim/ambient.wav',
    'C:/Users/Tze Lun/data prelim/data_prelim/free_running_drill.wav',
    'C:/Users/Tze Lun/data prelim/data_prelim/free_running_drill_2.wav',
    'C:/Users/Tze Lun/data prelim/data_prelim/pressing_1.wav',
    'C:/Users/Tze Lun/data prelim/data_prelim/pressing_2.wav',
    'C:/Users/Tze Lun/data prelim/data_prelim/pressing_3.wav'
]

an = {
    'ambient': 0,
    'free drilling soft': 1,
    'free drilling': 2,
    'drilling 1': 3,
    'drilling 2': 4,
    'drilling 3': 5
}


