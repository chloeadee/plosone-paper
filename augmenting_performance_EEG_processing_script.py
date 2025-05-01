"""
Individual alpha frequency and aperiodic exponent estimatation for Pasithea
Author: Zachariah R Cross, 2020; adapted by Chloe Dziego 2021.

"""

import mne
import glob
import yasa
import pandas as pd
import os.path as op
import seaborn as sns
import matplotlib.pyplot as plt
from philistine.mne import savgol_iaf

#Make the plots pretty
sns.set(style='white', font_scale=1.2)

#Read in the raw files
raw_files = glob.glob('data/Pasithea_RESTING/*.vhdr')

# Set parameters for EOG and reference
eye_channels = ['FP1','FP2']
ref_channels = ['TP9','TP10']

#Set which electrodes we want to use for IAF estimation
electrodes = ['P3','P4','O1','O2','P7','P8', 'Oz']

#Create .txt file to save IAF values
outfile = open('iaf_output.txt','w')
header = "subj"+"\t"+"measure"+"\t"+"value"+"\t"+"session"+"\n"
outfile.write(header)

#Set the montage
montage = 'standard_1020'

#Pre-Processing Loop
for r in raw_files:
	#Extract the participant information for exporting later
    print("processing file "+r)
    subj = op.split(r)[1][0:4] #extract first 3 characters (i.e., CP01)
    session = op.split(r)[1][6:7]
    
    #Read in the raw data
    raw = mne.io.read_raw_brainvision(r, eog=eye_channels, preload=True)    
	
	#Remove unneeded channels
    raw.drop_channels(['FP1', 'FP2', 'x_dir', 'y_dir', 'z_dir'])
    
    #Set the montage
    raw.set_montage(montage)
     
    #Re-reference our EEG to linked mastoids
    raw = mne.io.set_eeg_reference(raw,ref_channels)[0]
    
    #Set the unneeded channel types
    raw.set_channel_types({'TP9': 'misc', 'TP10': 'misc'})
        
    #Downsample to 250 Hz
    raw = raw.resample(250)
    
    #Apply a basic preprocessing step to the data i.e., filter from 1-40 Hz to tidy slightly
    raw = raw.filter(1, 40.,
                    l_trans_bandwidth='auto',
                     h_trans_bandwidth='auto',
                     filter_length='auto',
                     method='fir',
                     fir_window='hamming',
                     phase='zero',
                     n_jobs=2)

	#Choose the electrodes to save data from
    picks = list()
    for e in electrodes:
        index = raw.ch_names.index(e)
        picks.append(index)
        
#IAF estimation
    #Get peak alpha frequency (paf) and centre of gravity (cog) estimates
    paf, cog, ablimits = savgol_iaf(raw, picks=picks, fmin=7, fmax=13)

    #Write the paf and cog values into our .txt file that we created on
    #Line 38 and save it to our working directory
    outfile.write(subj+"\t"+"paf\t"+str(paf)+"\t"+session+"\n")
    outfile.write(subj+"\t"+"cog\t"+str(cog)+"\t"+session+"\n")
    
    #Finally, let's save the figure showing peak IAF for each subject
    plt.title('IAF at occipital-parietal channels' + ' for ' + subj)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectral Density ($uV^2$/Hz)')
    plt.savefig('iaf_figures/' + subj + '_' + session + '.png');
    plt.close();
    
    
#Aperiodic Estimation 
    #Set parameters for analysis
    data = raw.get_data()
    sf = raw.info['sfreq']
    chan = raw.ch_names
    
    #Let's check our channel list, sampling frequency and length of the data
    print('Chan =', chan)
    print('Sampling frequency =', sf, 'Hz')
    print('Data shape =', data.shape)
    print('Duration =', data.shape[1] / sf, 'seconds')
    
    #Now let's calculate the original power spectrum (PSD)
    from scipy.signal import welch
    
    win = int(4 * sf)                           # window size is 4 seconds
    freqs, psd = welch(data, sf, nperseg=win)   # single or multi-channel data
    
    print(freqs.shape, psd.shape)               # psd shape (n_chan, n_freq)
    
    #Now we can plot our PSD - 15 refers to the channel we are selecting (15 = Oz)
    #This is a good choice because we can easily see the peak alpha power.
    plt.plot(freqs, psd[15, :], 'k', lw=2.5)
    plt.fill_between(freqs, psd[15, :], cmap='Spectral')
    plt.xlim(1, 30)
    plt.yscale('log')
    sns.despine()
    plt.title(chan[15])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD log($uV^2$/Hz)')   
    plt.savefig('irasa_figures/' + subj + '_' + session + '_PSD.png')
    plt.close();
    
    #Apply the IRASA technique - this is the function to calculate and separate the aperiodic from the true oscillatory activity
    freqs, psd_aperiodic, psd_osc = yasa.irasa(data, sf, ch_names=chan, 
                                               band=(1, 30), win_sec=4, 
                                               return_fit=False)
    
    #Now let's plot the aperiodic component on a linear-log scale
    plt.plot(freqs, psd_aperiodic[15, :], 'k', lw=2.5)
    plt.fill_between(freqs, psd_aperiodic[15, :], cmap='Spectral')
    plt.xlim(1, 30)
    plt.yscale('log')
    sns.despine()
    plt.title('Aperiodic component at ' + chan[15])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD log($uV^2$/Hz)')
    plt.savefig('irasa_figures/' + subj + '_' + session + '_aperiodic.png')
    plt.close();
    
    #And oscillatory component on a linear-linear scale
    plt.plot(freqs, psd_osc[15, :], 'k', lw=2.5)
    plt.fill_between(freqs, psd_osc[15, :], cmap='Spectral')
    plt.xlim(1, 30)
    sns.despine()
    plt.title('Oscillatory component at ' + chan[15])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD log($uV^2$/Hz)')
    plt.savefig('irasa_figures/' + subj + '_' + session + '_oscillatory.png')
    plt.close();
    
    #Finally, let's fit the fractal component (1/f)
    #Here, we are fitting an exponential function to the aperiodic 
    #power spectrum and return the fit parameters (intercept, slope), the R^2 
    #of the fit, and the standard deviation of the oscillatory component.
    
    freqs, psd_aperiodic, psd_osc, fit_params = yasa.irasa(data, sf, 
                                                           ch_names=chan)
    fit_params
    
    #Add an additional column to include subject number
    fit_params['subj'] = subj
    fit_params['session'] = session
    
    #Save output file for each subject
    #Add to dataframe 
    fit_params.to_csv('irasa_data/' + subj + '_' + session + '_aperiodic.csv', header = True)
    
    df_export = pd.DataFrame(fit_params)
    if op.isfile('aperiodic.csv'):
        df_export.to_csv('aperiodic.csv', sep=',', mode='a', header=False)
    else:
        df_export.to_csv('aperiodic.csv', sep=',', mode='a', header=True)

#Close the data file
outfile.close()
print("Preprocessing Complete")
