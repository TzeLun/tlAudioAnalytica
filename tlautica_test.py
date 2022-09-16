import tlautica

path = 'C:/Users/Tze Lun/data prelim/data_prelim/free_running_drill_2.wav'

frdr = tlautica.TLAudioHandler(path)
frdr.figProperties['title'] = 'free running drill'
frdr.plot_waveform(annotation=False)
frdr.save_plot_as_image('hello', tight=True)
