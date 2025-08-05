import numpy as np

import data_utils
import plot_utils
import classifier
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def neural_population(this_structure_units_table, stimuli):
    n_neurons = len(this_structure_units_table.spike_times.values)
    stim_times = stimuli.start_time

    bins = np.arange(-.2,.5,.1)
    storage = np.empty((n_neurons,len(stim_times),len(bins)-1))

    for nn in range(n_neurons):
        spike_times = this_structure_units_table.spike_times.values[nn]

        spike_count = []
        trial_index = []

        storage[nn,:,:]  = get_binned_triggered_spike_counts_fast(spike_times,stim_times,bins)

    trial_index = np.arange(len(spike_count))
    trial_id_types, trial_id = np.unique(stimuli.image_name.values,return_inverse = True)

    return storage, trial_id


def main(unit, path):

    session, units_table, stimuli, good_units = data_utils.load_nwb(path)
    this_structure_units_table = good_units[good_units.location == 'VISp']

    # Single neuron 
    # Extract spike times and get the times that the stimulus presentation started for a neuron (unit)
    spike_times = this_structure_units_table.spike_times.values[unit]
    stim_times = stimuli.start_time.values
    spike_count, trial_id = data_utils.get_spike_counts(spike_times, stim_times, stimuli)

    # x = predictor, y = class 
    x = spike_count
    y = trial_id

    ### UNCOMMENT BELOW FOR NEURAL POPULATION DECONDING EXAMPLE

    # # Population of neurons 
    # # Extract spike times for a set of neurons
    # trial_id, storage = neural_population(this_structure_units_table, stimuli)
    # stimulus_change_number = stimuli.flashes_since_change.values
    # change_number = 0

    # # Select a time window to use for decoding
    # inc_time_idx = np.where((bins>=0) & (bins<.3))[0] # select times to include
    # start_idx = np.min(inc_time_idx)
    # end_idx = np.max(inc_time_idx)

    # # Find the number of spikes in the selected window
    # x = np.sum(storage[:,stimulus_change_number==change_number,start_idx:end_idx],axis=2).T 
    # # And the trial identity for each of the selected stimuli
    # y = trial_id[stimulus_change_number==change_number]

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y) 

    # Classifier
    svc = classifier.fit_classifier(x_train, y_train)
    y_prediction, score = classifier.run_classifier(svc, x_test, y_test)
    return y_prediction, score


if __name__ == "__main__":

    example_sessions = [1139846596, 1152811536, 1069461581]
    this_session = str(example_sessions[0])
    this_filename = f'ecephys_session_{this_session}.nwb'

    parser = argparse.ArgumentParser(description="Run a classifier")
    parser.add_argument("--path_to_data", default=f"visual-behavior-neuropixels'/'behavior_ecephys_sessions'/{this_session}/{this_filename}", help="Path to data with behavior ecephys sessions")

    args = parser.parse_args()

    path = args.path_to_data
    unit = 80

    main(unit, path)


    