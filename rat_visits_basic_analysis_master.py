# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:44:39 2023

@author: ashutoshshukla
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import itertools
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

#%% Function to read Excel files in a folder
def read_excel_files_in_folder(folder_path):
    
    
    dataframes = {}
    
    excel_files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx') and not file.startswith('~$')]

    if not excel_files:
        print(f"No Excel files found in the folder: {folder_path}")
        return dataframes

    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        excel_data = pd.read_excel(file_path, sheet_name=None)
        
        for sheet_name, df in excel_data.items():
            key = f"{os.path.splitext(file)[0]} - {sheet_name}"
            dataframes[key] = df

    return dataframes

#%% Specify the root folder containing the subfolders with Excel files
root = tk.Tk()
root.withdraw()

folder_path = filedialog.askdirectory(title="Select Folder Containing Excel Files")
if not folder_path:
    print("No folder selected.")

#%% Helper function to identify transitions in a well_entries series

def identify_transitions(series):
    return (series != series.shift()).astype(int)

#%% Lempel_Ziv complexity (From Phillip Faure's paper)

import numpy as np

# Calculate the Lempel-Ziv (LZ) complexity of a sequence
def calculate_lz_complexity(sequence):
    substrings = {}
    complexity = 0
    substring = ""
    
    for symbol in sequence:
        substring += str(symbol)
        if substring not in substrings:
            substrings[substring] = True
            complexity += 1
            substring = ""
    
    return complexity

#%% Helper function for calculating normalized LZ-complexity

# Calculate normalized LZ-complexity (NLZcomp)
def calculate_nlz_complexity(sequence, num_surrogates=1000):
    lz_complexity = calculate_lz_complexity(sequence)
    surrogate_complexities = []

    for _ in range(num_surrogates):
        surrogate_sequence = np.random.choice(sequence, size=len(sequence), replace=True)
        surrogate_complexities.append(calculate_lz_complexity(surrogate_sequence))
    
    avg_surrogate_complexity = np.mean(surrogate_complexities)
    nlz_complexity = lz_complexity / avg_surrogate_complexity

    return nlz_complexity


#%% Helper function to calculate moving average

# Function to calculate the moving average
def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=2).mean()

#%% Get a list of all subfolders in the root folder
subfolders = [os.path.join(folder_path, folder) for folder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, folder))]

# Define a color mapping for subfolders (assign unique colors)
subfolder_colors = {
    'Pair_1 3': 'red',
    'Pair_2 4': 'black'
    }

num_wells = 3
epsilon = 1e-10

result_dataframes = {}  # Create an empty dictionary to store processed dataframes
rat1_rewards_dict = {}  # Dictionary to store rat1 rewards
rat2_rewards_dict = {}  # Dictionary to store rat2 rewards

rat1_matches_dict = {}  # Dictionary to store rat1 matches
rat2_matches_dict = {}  # Dictionary to store rat2 matches

rat1_reward_intervals_dict = {}  # Dictionary to store rat1 reward intervals
rat2_reward_intervals_dict = {}  # Dictionary to store rat2 reward intervals

rat1_mean_reward_interval_dict = {}  # Dictionary to store rat1 reward intervals
rat2_mean_reward_interval_dict = {}  # Dictionary to store rat2 reward intervals

rat1_match_intervals_dict  ={} # Dictionary to store rat1 match intervals
rat2_match_intervals_dict  ={} # Dictionary to store rat2 match intervals

rat1_mean_match_interval_dict  ={} # Dictionary to store rat1 match intervals
rat2_mean_match_interval_dict  ={} # Dictionary to store rat2 match intervals

rat1_transitions_dict = {}  # Dictionary to store rat1 transitions
rat2_transitions_dict = {}  # Dictionary to store rat2 transitions

rat1_tpm_dict = {}  # Dictionary to store rat1 transitions
rat2_tpm_dict = {}  # Dictionary to store rat2 transitions

lz_complexity_rat1 = {}
lz_complexity_rat2 = {}

nlz_complexity_rat1 = {}
nlz_complexity_rat2 = {}

rqa_entropy_rat1 = {}
rqa_entropy_rat2 = {}

tau_dict = {}
tau_p_value_dict = {}

# Iterate through subfolders and analyze data in each folder
for folder in subfolders:
    print(f"Analyzing data in folder: {folder}")
    dataframes = read_excel_files_in_folder(folder)
    
    # Get the subfolder ID from the folder path
    subfolder_id = os.path.basename(folder)
    
    # Get the color for this subfolder based on the mapping
    subfolder_color = subfolder_colors.get(subfolder_id, subfolder_colors)  # Default to black if not found
    
    if dataframes:
      # Sort the keys in ascending order
      sorted_keys = list(dataframes.keys())

    

    # Process pairs of keys
    for key_idx in range(0, len(sorted_keys), 2):
        key = sorted_keys[key_idx]
        next_key = sorted_keys[key_idx + 1] if key_idx + 1 < len(sorted_keys) else None

        print(f"Processing data for keys: {key} and {next_key}")

        rat1_dataframe = dataframes[key]
        rat2_dataframe = dataframes[next_key] if next_key is not None else None
       
        # Combine timestamps from both dataframes and remove duplicates
        all_timestamps = np.unique(np.concatenate((rat1_dataframe['start'], rat2_dataframe['start'])))

        # Create a common timeline with evenly spaced timestamps
        common_timeline = pd.DataFrame({'common_start': all_timestamps})
        common_timeline.set_index('common_start', inplace=True)

        # Resample and align data onto the common timeline
        rat1_dataframe_resampled = rat1_dataframe.set_index('start').reindex(common_timeline.index, method='nearest')
        rat2_dataframe_resampled = rat2_dataframe.set_index('start').reindex(common_timeline.index, method='nearest')
       
        # Calculate Kendall's Tau
        tau, p_value = kendalltau(rat1_dataframe_resampled['thiswell'], rat2_dataframe_resampled['thiswell'])
       
     
        # Calculate LZ complexity for rat 1
        lz_complexity_1 = calculate_lz_complexity(rat1_dataframe['thiswell'])
        nlz_complexity1 = calculate_nlz_complexity(rat1_dataframe['thiswell'])
       
        # Calculate LZ complexity for rat 2
        lz_complexity_2 = calculate_lz_complexity(rat2_dataframe['thiswell'])
        nlz_complexity2 = calculate_nlz_complexity(rat2_dataframe['thiswell'])
       
        # # Create a RecurrencePlot instance for rat 1
        # rat1_rp = RecurrencePlot(rat1_dataframe['thiswell'])
        # # Get the diagonal lines and calculate RQA ENT
        # rat1_diagonal_lines = rat1_rp.diagonal_lines()
        # rat1_diagonal_length_frequencies = [len(line) for line in rat1_diagonal_lines]
        # rat1_total_diagonal_lengths = sum(rat1_diagonal_length_frequencies)

        # # Calculate the frequency distribution of diagonal lengths
        # rat1_diagonal_length_probabilities = [length / rat1_total_diagonal_lengths for length in rat1_diagonal_length_frequencies]

        # # Calculate entropy based on diagonal length probabilities
        # rat1_rqa_entropy = -sum(p * np.log2(p) for p in rat1_diagonal_length_probabilities if p != 0)

        # # Create a RecurrencePlot instance for rat 1
        # rat2_rp = RecurrencePlot(rat2_dataframe['thiswell'])
        # # Get the diagonal lines and calculate RQA ENT
        # rat2_diagonal_lines = rat2_rp.diagonal_lines()
        # rat2_diagonal_length_frequencies = [len(line) for line in rat2_diagonal_lines]
        # rat2_total_diagonal_lengths = sum(rat2_diagonal_length_frequencies)

        # # Calculate the frequency distribution of diagonal lengths
        # rat2_diagonal_length_probabilities = [length / rat2_total_diagonal_lengths for length in rat2_diagonal_length_frequencies]

        # # Calculate entropy based on diagonal length probabilities
        # rat2_rqa_entropy = -sum(p * np.log2(p) for p in rat2_diagonal_length_probabilities if p != 0)
       

        rat1_rewards = rat1_dataframe['Reward'].sum()
        print(f"Rat1 obtained {rat1_rewards} rewards")

        rat2_rewards = rat2_dataframe['Reward'].sum()
        print(f"Rat2 obtained {rat2_rewards} rewards")

        rat1_matches = rat1_dataframe['match'].sum()
        print(f"Rat1 matched {rat1_matches} times")

        rat2_matches = rat2_dataframe['match'].sum()
        print(f"Rat2 matched {rat2_matches} times")

        # Calculate the interval between successive rewards for rat1
        rat1_reward_times = rat1_dataframe[rat1_dataframe['Reward'] == 1]['start']
        rat1_reward_intervals = rat1_reward_times.diff().dropna()
        average_interval_rat1 = rat1_reward_intervals.mean()
        # print(f"Average interval between rat1 rewards: {average_interval_rat1}")
           
        # Calculate the interval between successive matches for rat1
        rat1_match_times = rat1_dataframe[rat1_dataframe['match'] == 1]['start']
        rat1_match_intervals = rat1_match_times.diff().dropna()
        average_match_interval_rat1 = rat1_match_intervals.mean()
        # print(f"Average interval between rat1 matches: {average_match_interval_rat1}")
           
        # Calculate transitions for rat1
        rat1_transitions = (rat1_dataframe['thiswell'] != rat1_dataframe['thiswell'].shift()).sum()
           
        # Calculate transitions for rat1
        rat1_transits = identify_transitions(rat1_dataframe['thiswell'])
           
        # Calculate transition matrices for rat1
        transition_matrix_rat1 = np.zeros((num_wells, num_wells))
        for k in range(len(rat1_dataframe['thiswell']) - 1):
            from_state_rat1 = int(rat1_dataframe['thiswell'].iloc[k]) - 1
            to_state_rat1 = int(rat1_dataframe['thiswell'].iloc[k + 1]) - 1
            transition_matrix_rat1[from_state_rat1, to_state_rat1] += 1
            transition_prob_matrix_rat1 = transition_matrix_rat1 / (transition_matrix_rat1.sum(axis=1, keepdims=True) + epsilon)
               

        # Calculate the interval between successive rewards for rat2
        if rat2_dataframe is not None:
            rat2_reward_times = rat2_dataframe[rat2_dataframe['Reward'] == 1]['start']
            rat2_reward_intervals = rat2_reward_times.diff().dropna()
            average_interval_rat2 = rat2_reward_intervals.mean()
            # print(f"Average interval between rat2 rewards: {average_interval_rat2}")
               
            # Calculate the interval between successive matches for rat1
            rat2_match_times = rat2_dataframe[rat2_dataframe['match'] == 1]['start']
            rat2_match_intervals = rat2_match_times.diff().dropna()
            average_match_interval_rat2 = rat2_match_intervals.mean()
            # print(f"Average interval between rat2 matches: {average_match_interval_rat2}")
               
            # Calculate transitions for rat1
            rat2_transitions = (rat2_dataframe['thiswell'] != rat2_dataframe['thiswell'].shift()).sum()
               
            # Calculate transitions for rat1
            rat2_transits = identify_transitions(rat2_dataframe['thiswell'])
               
            # Calculate transition matrices for rat1
            transition_matrix_rat2 = np.zeros((num_wells, num_wells))
            for k in range(len(rat2_dataframe['thiswell']) - 1):
                from_state_rat2 = int(rat2_dataframe['thiswell'].iloc[k]) - 1
                to_state_rat2 = int(rat2_dataframe['thiswell'].iloc[k + 1]) - 1
                transition_matrix_rat2[from_state_rat1, to_state_rat2] += 1
                transition_prob_matrix_rat2 = transition_matrix_rat2 / (transition_matrix_rat2.sum(axis=1, keepdims=True) + epsilon)


                # Merge and process data as before
                data = pd.merge(rat1_dataframe, rat2_dataframe, on='start', how='outer')
                data.fillna(method='ffill', inplace=True)
                data = data.dropna()
                result_dataframes[key] = data

                # Store rewards and matches in dictionaries
                rat1_rewards_dict[key] = rat1_rewards
                rat2_rewards_dict[key] = rat2_rewards
               
                rat1_matches_dict[key] = rat1_matches
                rat2_matches_dict[key] = rat2_matches
               
                rat1_reward_intervals_dict[key] = rat1_reward_intervals
                rat2_reward_intervals_dict[key] = rat2_reward_intervals
               
                rat1_mean_reward_interval_dict[key] = average_interval_rat1
                rat2_mean_reward_interval_dict[key] = average_interval_rat2
               
                rat1_match_intervals_dict[key]  = rat1_match_intervals
                rat1_match_intervals_dict[key] = rat2_match_intervals
               
                rat1_mean_match_interval_dict[key] = average_match_interval_rat1
                rat2_mean_match_interval_dict[key] = average_match_interval_rat2
               
                rat1_transitions_dict[key] = rat1_transitions
                rat2_transitions_dict[key] = rat2_transitions
               
                rat1_tpm_dict[key] = transition_prob_matrix_rat1
                rat2_tpm_dict[key] = transition_prob_matrix_rat2
               
                lz_complexity_rat1[key] = lz_complexity_1
                lz_complexity_rat2[key] = lz_complexity_2
               
                nlz_complexity_rat1[key] = nlz_complexity1
                nlz_complexity_rat2[key] = nlz_complexity2
               
                # rqa_entropy_rat1[key] =  rat1_rqa_entropy
                # rqa_entropy_rat2[key] =  rat2_rqa_entropy
               
               
                tau_dict[key] = tau
                tau_p_value_dict[key] = p_value
               

    
    # Plot rewards for rat1
    plt.figure(figsize=(10, 6))
    rewards_values = [rat1_rewards_dict[key] for key in rat1_rewards_dict.keys()]

    # Calculate the moving average with a window size of 5 (adjust as needed)
    rewards_values = moving_average(pd.Series(rewards_values, dtype = 'float64'), window_size=2)

    matches_values = [rat1_matches_dict[key] for key in rat1_matches_dict.keys()]

    # Calculate the moving average with a window size of 5 (adjust as needed)
    matches_values = moving_average(pd.Series(matches_values, dtype = 'float64'), window_size=2)

    # Create a single figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot( rewards_values, marker='o', color = subfolder_color, label='Rewards')
    ax1.set_xlabel('Run')
    ax1.set_ylabel('# Rewards', color=subfolder_color)
    ax1.tick_params(axis='y', labelcolor=subfolder_color)
    ax1.set_ylim(-5, 100)


    ax2 = ax1.twinx()

    ax2.plot(matches_values, marker='x', color = subfolder_color, linestyle = '--', label='Matches')
    ax2.set_ylabel('# Matches', color=subfolder_color)
    ax2.tick_params(axis='y', labelcolor=subfolder_color)
    ax2.set_ylim(-5, 100)
    
    # Plot the reward/match ratio
    # Plot rewards for rat1
    plt.figure(figsize=(10, 6))

    reward_match_ratio = rewards_values/matches_values


    # Create a single figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot( reward_match_ratio, marker='o', color = subfolder_color, label='Rewards')
    ax1.set_xlabel('Run')
    ax1.set_ylabel('Rewards / match', color=subfolder_color)
    ax1.set_ylim(0, 1.2)
    
    plt.figure(figsize=(10, 6))
    rat1_transition_values = [rat1_transitions_dict[key] for key in rat1_transitions_dict.keys()]

    # Calculate the moving average with a window size of 5 (adjust as needed)
    rat1_transition_values = moving_average(pd.Series(rat1_transition_values, dtype = 'float64'), window_size=5)

    rat2_transition_values = [rat2_transitions_dict[key] for key in rat2_transitions_dict.keys()]

    # Calculate the moving average with a window size of 5 (adjust as needed)
    rat2_transition_values = moving_average(pd.Series(rat2_transition_values, dtype = 'float64'), window_size=5)

    plt.plot(rat1_transition_values, marker='o', color=subfolder_color)
    plt.plot(rat2_transition_values, marker='x', color=subfolder_color, linestyle = '--')
    plt.xlabel('Run')
    plt.ylabel('# Transitions')

    # Plot the reward/match ratio

    # Plot rewards for rat1
    plt.figure(figsize=(10, 6))

    reward_match_ratio = rewards_values/matches_values


    # Create a single figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot( reward_match_ratio, marker='o', color=subfolder_color, label='Rewards')
    ax1.set_xlabel('Run')
    ax1.set_ylabel('Rewards / match', color=subfolder_color)
    ax1.set_ylim(0, 1.2)
    
    # Plot the behavioral performances normalized by the number of transitions made by the pair

    # Plot rewards for rat1
    plt.figure(figsize=(10, 6))

    reward_metric = rewards_values/(rat1_transition_values + rat2_transition_values)
    matches_metric = matches_values/ (rat1_transition_values + rat2_transition_values)



    # Create a single figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot( reward_metric, marker='o', color=subfolder_color, label='Rewards')
    ax1.set_xlabel('Run')
    ax1.set_ylabel('Rewards / transition', color=subfolder_color)
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, 0, 0.67, color = subfolder_color, linestyle = '--')
    ax1.axhline(0.25, 0.67, 1.0, color = subfolder_color, linestyle = '--')
    ax1.tick_params(axis='y', labelcolor=subfolder_color)


    ax2 = ax1.twinx()

    ax2.plot(matches_metric, marker='x', color=subfolder_color, linestyle = '--', label='Matches')
    ax2.set_ylabel('Matches / transition', color=subfolder_color)
    ax2.tick_params(axis='y', labelcolor=subfolder_color)
    ax2.set_ylim(0, 1)
    
    # Plot mean inter-reward intervals for rats 1 and 2

    # Plot rewards for rat1
    plt.figure(figsize=(10, 6))
    rat1_rewards_intervals = [rat1_mean_reward_interval_dict[key] for key in rat1_mean_reward_interval_dict.keys()]

    # Calculate the moving average with a window size of 5 (adjust as needed)
    rat1_rewards_intervals = moving_average(pd.Series(rat1_rewards_intervals), window_size=5)

    rat2_rewards_intervals = [rat2_mean_reward_interval_dict[key] for key in rat2_mean_reward_interval_dict.keys()]


    # Calculate the moving average with a window size of 5 (adjust as needed)
    rat2_rewards_intervals = moving_average(pd.Series(rat2_rewards_intervals), window_size=5)

    # Create a single figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot( rat1_rewards_intervals, marker='o', color=subfolder_color, label='Rewards')
    ax1.set_xlabel('Run')
    ax1.set_ylabel('Rat 1 inter-reward interval', color=subfolder_color)
    ax1.tick_params(axis='y', labelcolor=subfolder_color)

    ax2 = ax1.twinx()

    ax2.plot(rat2_rewards_intervals, marker='x', color=subfolder_color, linestyle = '--', label='Matches')
    ax2.set_ylabel('Rat 2 inter-reward interval', color=subfolder_color)
    ax2.tick_params(axis='y', labelcolor=subfolder_color)


    # Plot mean inter-match intervals for rats 1 and 2

    # Plot rewards for rat1
    plt.figure(figsize=(10, 6))
    rat1_match_intervals = [rat1_mean_match_interval_dict[key] for key in rat1_mean_match_interval_dict.keys()]

    # Calculate the moving average with a window size of 5 (adjust as needed)
    rat1_match_intervals = moving_average(pd.Series(rat1_match_intervals), window_size=5)

    rat2_match_intervals = [rat2_mean_match_interval_dict[key] for key in rat2_mean_match_interval_dict.keys()]

    # Calculate the moving average with a window size of 5 (adjust as needed)
    rat2_match_intervals = moving_average(pd.Series(rat2_match_intervals), window_size=5)

    # Create a single figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot( rat1_match_intervals, marker='o', color=subfolder_color, label='Rewards')
    ax1.set_xlabel('Run')
    ax1.set_ylabel('Rat 1 inter-match interval', color=subfolder_color)
    ax1.tick_params(axis='y', labelcolor=subfolder_color)
    ax1.set_ylim(0, 1700)

    ax2 = ax1.twinx()

    ax2.plot(rat2_match_intervals, marker='x', color=subfolder_color, label='Matches')
    ax2.set_ylabel('Rat 2 inter-match interval', color=subfolder_color)
    ax2.tick_params(axis='y', labelcolor=subfolder_color)
    ax2.set_ylim(0, 1700)

    # PLot LZ-complexity of choice sequences

    # Plot rewards for rat1
    plt.figure(figsize=(10, 6))
    lz1_values = [lz_complexity_rat1[key] for key in lz_complexity_rat1.keys()]

    # Calculate the moving average with a window size of 5 (adjust as needed)
    lz1_values = moving_average(pd.Series(lz1_values), window_size=5)

    lz2_values = [lz_complexity_rat2[key] for key in lz_complexity_rat2.keys()]

    # Calculate the moving average with a window size of 5 (adjust as needed)
    lz2_values = moving_average(pd.Series(lz2_values), window_size=5)


    # Create a single figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot( lz1_values, marker='o', color = subfolder_color)
    ax1.plot( lz2_values, marker='x', color = subfolder_color)
    ax1.set_xlabel('Run')
    ax1.set_ylabel('LZ-complexity')


    # Plot normalized LZ-complexity
    plt.figure(figsize=(10, 6))
    nlz1_values = [nlz_complexity_rat1[key] for key in nlz_complexity_rat1.keys()]

    # Calculate the moving average with a window size of 5 (adjust as needed)
    nlz1_values = moving_average(pd.Series(nlz1_values), window_size=5)

    nlz2_values = [nlz_complexity_rat2[key] for key in nlz_complexity_rat2.keys()]

    # Calculate the moving average with a window size of 5 (adjust as needed)
    nlz2_values = moving_average(pd.Series(nlz2_values), window_size=5)


    # Create a single figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(nlz1_values, marker='o', color = subfolder_color)
    ax1.plot(nlz2_values, marker='x', color = subfolder_color)
    ax1.axhline(1, color = subfolder_color, linestyle = '--')
    ax1.set_ylim(0.5, 1.1)
    ax1.set_xlabel('Run')
    ax1.set_ylabel('Normalized LZ-complexity')
