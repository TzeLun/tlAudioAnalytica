import tlautica

path = 'C:/Users/Tze Lun/data prelim/data_prelim/free_running_drill_2.wav'

frdr = tlautica.TLAudioHandler(path)
frdr.figProperties['title'] = 'free running drill'
frdr.set_power(2.0)
frdr.segment_audio(4.5, 0.1)
frdr.set_segment_as_main()
frdr.resample(22050)
frdr.plot_waveform(annotation=False)
frdr.plot_mel_spectrogram(True)
frdr.save_waveform_as_image('hello3', size=(1024, 1024), annotation=True, tight=False)
frdr.save_mel_spectrogram_as_image('hello5', size=(256, 256), annotation=False, tight=True)