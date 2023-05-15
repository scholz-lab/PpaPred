import pywt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_ave_values(xvalues, yvalues, n = 5):
    signal_length = len(xvalues)
    if signal_length % n == 0:
        padding_length = 0
    else:
        padding_length = n - signal_length//n % n
    xarr = np.array(xvalues)
    yarr = np.array(yvalues)
    xarr.resize(signal_length//n, n)
    yarr.resize(signal_length//n, n)
    xarr_reshaped = xarr.reshape((-1,n))
    yarr_reshaped = yarr.reshape((-1,n))
    x_ave = xarr_reshaped[:,0]
    y_ave = np.nanmean(yarr_reshaped, axis=1)
    return x_ave, y_ave

def cwt_signal(signal, scales, dt=1/30, 
                 waveletname = 'gaus5', #cgau5
              ):
    """scales: defines the width of a wavelet. Related to frequency of waves. Since waves have diverse forms and are located in time, it is better to speak of pseudo frequencies
    scales are inversly related to frequencies. the fuction pywt.scale2frequency can be used to determine the frequency. But returns frequency for sampling period -> *30 if 30fps -> Hz
    depending on wavelet and dt, scales lead to different frequencies: for gaus5 and 30 fps scale of 1 -> 15 Hz, scale 10 -> 1.5 Hz
    scales: array like
    signal: array like
    
    returns coefficients in imaginary numbers, make absolute to compute further with
    """
    coefficients, frequencies = pywt.cwt(signal, scales, waveletname, dt)
    
    return coefficients, frequencies

def plot_wavelet(coefficients, frequencies, 
                 cmap = plt.cm.magma, 
                 title = 'Wavelet Transform (Power Spectrum) of signal with cgau5', 
                 ylabel = 'Pseudo-Frequency (frames)', 
                 xlabel = 'Time'
                ):
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.25,  0.5,  1.,  4.,  8, 16., 32.]
    contourlevels = np.log2(levels)
    
    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    
    yticks = 2**np.arange(np.log2(period.min()), np.log2(period.max()))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(scales)
    ax.invert_yaxis()
    #ylim = ax.get_ylim()
    #ax.set_ylim(ylim[0], 0)
    
    #cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, orientation="vertical")#cax=cbar_ax
    plt.show()

def plot_signal_plus_average(time, signal, average_over = 5):
    time = np.arange(0, signal.shape[0])
    fig, ax = plt.subplots(figsize=(15, 3))
    time_ave, signal_ave = get_ave_values(time, signal, average_over)
    ax.plot(time, signal, label='signal')
    ax.plot(time_ave, signal_ave, label = 'time average (n={})'.format(5))
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('Signal Amplitude', fontsize=18)
    ax.set_title('Signal + Time Average', fontsize=18)
    ax.set_xlabel('Time', fontsize=18)
    ax.legend()
    plt.show()
    
def lowpassfilter(signal, thresh = 0.63, wavelet="db8"):
    """signal: array like, without nan values
    """
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal