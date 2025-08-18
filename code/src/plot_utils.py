import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix

def create_raster(
    spike_times,
    stim_index,
    ax,
    size=1,
    color='black'
):
    """
    Create a raster plot.
    Args:
        spike_times (numpy.ndarray): Times of all spikes (e.g. in seconds).
        stim_index (numpy.ndarray): Times of stimulus onsets.
        ax (matplotlib.axes.Axes): Axes on which to generate plot.
        size (int): Size of scatter plot point.
        color (string): Color of scatter plot point.
    """
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
        """
    Create a peristimulus time histogram.
    Args:
        spike_times (numpy.ndarray): Times of all spikes (e.g. in seconds).
        stim_index (numpy.ndarray): Times of stimulus onsets.
        ax (matplotlib.axes.Axes): Axes on which to generate plot.
        pre_window (float): How far before the stimulus to look.
        post_window (float): How far after the stimulus to look.
        bin_size (float): Size of the bins for the PSTH.
        color (string): Color of scatter plot point.
        label (string): Label for plot group if generating a legend.
    """
    # Set up bins
    bins = np.arange(-pre_window,post_window+bin_size,bin_size) 
    bin_centers = bins[:-1] + bin_size/2  

    a, b = np.histogram(spike_times, bins=bins)

    # Divide by # of trials, then bin size to get a rate estimate in Spikes/Sec = Hz
    a = a/np.max(stim_index)/bin_size
    ax.plot(bin_centers, a, c=color, label=label)
    ax.set_xlabel('Time from stimulus (seconds)')
    ax.set_ylabel('Spike Rate (Hz)')

def create_confusion_matrix(
    ax,
    y_pred,
    y_test,
):
    """
    Create a confusion matrix given predictions from a classifier.
    Args:
        ax (matplotlib.axes.Axes): Axes on which to generate plot.
        y_pred (numpy.ndarray): Predicted classes for each test datapoint.
        y_test (numpy.ndarray): Target outputs for test data.
    """ 
    im  = ax.imshow(confusion_matrix(y_true=y_test, y_pred=y_pred, normalize='pred'))
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    cbar = plt.colorbar(im)
    cbar.set_label('Fraction Guessed')