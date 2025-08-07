# Load in packages
import numpy as np
import pandas as pd
import pynwb

# Set the data root according to OS
import platform
from pathlib import Path

platstring = platform.platform()

if 'Darwin' in platstring:
    # macOS 
    data_root = Path("/Volumes/Brain2025/")
elif 'Windows'  in platstring:
    # Windows (replace with the drive letter of USB drive)
    data_root = Path("E:/")
elif ('amzn' in platstring):
    # then on CodeOcean
    data_root = Path("/data/")
else:
    # then your own linux platform
    # EDIT location where you mounted hard drive
    data_root = Path("/media/$USERNAME/Brain2025/")


# Load in a session from a NWB file
def load_nwb(
    path, 
    max_isi_violations=0.5, 
    max_amplitude_cutoff=0.1, 
    min_presence_ratio=0.95
):
    """
    Load a NWB file and return the session, units table, and stimuli information.
    Args:
        path (str): Path to the NWB file (excludes the data root).
        max_isi_violations (float): Maximum allowed ISI violations for units.
        max_amplitude_cutoff (float): Maximum allowed amplitude cutoff for units.
        min_presence_ratio (float): Minimum presence ratio for units.
    Returns:
        session (pynwb.NWBFile): The loaded NWB session.
        units_table (pd.DataFrame): DataFrame containing units information.
        stimuli (pd.DataFrame): DataFrame containing stimuli information.
        good_units (pd.DataFrame): DataFrame containing units that meet the QC criteria.
    """

    nwb_path = data_root / path
    session = pynwb.NWBHDF5IO(nwb_path).read()
    
    # Get stimuli information
    stimuli = session.intervals['Natural_Images_Lum_Matched_set_ophys_H_2019_presentations'].to_dataframe()

    # Get units table
    units_table = session.units.to_dataframe()
    electrodes_table = session.electrodes.to_dataframe()
    units_electrode_table = units_table.join(electrodes_table,on = 'peak_channel_id')

    # Filter units by QC criteria
    good_units = units_electrode_table[
        (units_electrode_table.isi_violations < max_isi_violations) &
        (units_electrode_table.amplitude_cutoff < max_amplitude_cutoff) &
        (units_electrode_table.presence_ratio > min_presence_ratio)
    ]
    assert len(good_units) > 0, "There are 0 units that meet the specified QC criteria in this session."

    return session, units_table, stimuli, good_units

def get_stim_window(
    spike_times,
    stim_times,
    pre_window=0.2,     # How far before the stimulus should we look?
    post_window=0.75,   # How far after the stimulus should we look?
):
    
    # Storage for data
    triggered_spike_times = []
    triggered_stim_index = []

    # Loop through the stimuli
    for i, stim_time in enumerate(stim_times):
        # Select spikes that fall within the time window around this stimulus
        mask = ((spike_times >= stim_time - pre_window) & 
                (spike_times < stim_time + post_window))
        
        # Align spike times to stimulus onset (0 = stimulus)
        trial_spikes = spike_times[mask] - stim_time

        triggered_spike_times.append(trial_spikes)
        triggered_stim_index.append(np.ones(len(trial_spikes))*i)

    # For plotting, we are going to want to concatenate these data into one big vector
    triggered_spike_times = np.concatenate(triggered_spike_times)
    triggered_stim_index = np.concatenate(triggered_stim_index)

    return triggered_spike_times, triggered_stim_index

def get_spike_counts(
    spike_times,
    stim_times,
    stimuli,
    start=0,
    stop=0.35
):
    spike_count = []

    for i, stim_time in enumerate(stim_times):
        # Select spikes that fall within the time window around this stimulus
        mask = ((spike_times >= stim_time + start) & 
                (spike_times < stim_time + stop))
        
        # Count spikes in this bin
        spike_count.append(len(spike_times[mask]))
        
    spike_count = np.array(spike_count)
    _, trial_id = np.unique(stimuli.image_name.values, return_inverse= True)

    return spike_count, trial_id

# For poopulation decoding
def get_binned_triggered_spike_counts_fast(
    spike_times, 
    stim_times, 
    bins):
    """
    Fast peri-stimulus time histogram using searchsorted.

    Parameters
    ----------
    spike_times : 1D array_like, sorted
        Times of all spikes (e.g. in seconds).
    stim_times : 1D array_like
        Times of stimulus onsets.
    bins : 1D array_like
        Bin edges *relative* to stimulus (e.g. np.linspace(-0.1, 0.5, 61)).

    Returns
    -------
    counts : 2D ndarray, shape (n_trials, len(bins)-1)
        counts[i, j] is the number of spikes in bin j of trial i.
    """
    # ensure numpy arrays
    spike_times = np.asarray(spike_times)
    stim_times = np.asarray(stim_times)
    bins = np.asarray(bins)

    spike_times = np.sort(spike_times)

    n_trials = stim_times.size
    n_bins = bins.size - 1
    counts = np.zeros((n_trials, n_bins), dtype=int)

    for i, stim in enumerate(stim_times):
        # compute the absolute edges for this trial
        edges = stim + bins
        # find the insertion indices for each edge
        idx = np.searchsorted(spike_times, edges, side='left')
        # differences between successive indices = counts per bin
        counts[i, :] = np.diff(idx)

    return counts

def get_spike_counts_all(units_table, stim_times, bins):
    n_neurons = len(units_table.spike_times.values)
    spike_count = np.empty((n_neurons, len(stim_times), len(bins)-1))

    for nn in range(n_neurons):
        spike_times = units_table.spike_times.values[nn]
        spike_count[nn,:,:] = get_binned_triggered_spike_counts_fast(spike_times, stim_times, bins)
    return spike_count