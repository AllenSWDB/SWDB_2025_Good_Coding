import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

def create_raster(
    spike_times,
    stim_index,
    ax,
    size=1,
    color='black'
):
    ax.scatter(spike_times, stim_index, s=size,c=color)
    ax.set_xlabel('Time from stimulus (seconds)')
    ax.set_ylabel('Stim number (sorted)')
    ax.axvline([0],c = 'r')

def create_psth(
    spike_times,
    stim_index,
    ax,
    pre_window=0.2,     # How far before the stimulus should we look?
    post_window=0.75,   # How far after the stimulus should we look?
    bin_size=0.01,      # What size bins do we want for our PSTH?
    color='black',
    label=None
):
    # Set up bins
    bins = np.arange(-pre_window,post_window+bin_size,bin_size) 
    bin_centers = bins[:-1] + bin_size/2  

    a, b = np.histogram(spike_times, bins=bins)

    # Divide by # of trials, then bin size to get a rate estimate in Spikes/Sec = Hz
    a = a/np.max(stim_index)/bin_size
    ax.plot(bin_centers, a, c=color, label=label)
    ax.set_xlabel('Time from stimulus (seconds)')
    ax.set_ylabel('Spike Rate (Hz)')