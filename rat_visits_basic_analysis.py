# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:58:05 2023

@author: ashutoshshukla
"""

#%% Import required libraries

import os
import tkinter as tk
from tkinter import filedialog
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator

#%% Open excel files

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def read_excel_files_in_folder():
    root = tk.Tk()
    root.withdraw()

    folder_path = filedialog.askdirectory(title="Select Folder Containing Excel Files")
    if not folder_path:
        print("No folder selected.")
        return {}

    excel_files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx') and not file.startswith('~$')]

    if not excel_files:
        print("No Excel files found in the folder.")
        return {}

    dataframes = {}
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        excel_data = pd.read_excel(file_path, sheet_name=None)
        
        for sheet_name, df in excel_data.items():
            dataframes[f"{os.path.splitext(file)[0]} - {sheet_name}"] = df

    return dataframes

dataframes = read_excel_files_in_folder()


#%% Helper function to identify transitions in a well_entries series

def identify_transitions(series):
    return (series != series.shift()).astype(int)

#%% Extract data of interest for saving and plotting

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


if dataframes:
    # Sort the keys in ascending order
    sorted_keys = list(dataframes.keys())

    for key in sorted_keys:
        print(f"Processing data for key: {key}")

        # Process pairs of keys
        for key_idx in range(0, len(sorted_keys), 2):
            key = sorted_keys[key_idx]
            next_key = sorted_keys[key_idx + 1] if key_idx + 1 < len(sorted_keys) else None

            print(f"Processing data for keys: {key} and {next_key}")

            rat1_dataframe = dataframes[key]
            rat2_dataframe = dataframes[next_key] if next_key is not None else None

            rat1_rewards = rat1_dataframe['Reward'].sum()
            # print(f"Rat1 obtained {rat1_rewards} rewards")

            rat2_rewards = rat2_dataframe['Reward'].sum()
            # print(f"Rat2 obtained {rat2_rewards} rewards")

            rat1_matches = rat1_dataframe['match'].sum()
            # print(f"Rat1 matched {rat1_matches} times")

            rat2_matches = rat2_dataframe['match'].sum()
            # print(f"Rat2 matched {rat2_matches} times")

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
                
#%% Plot the rewards obtained, matches and transitions

# Plot rewards for rat1
plt.figure(figsize=(10, 6))
rewards_values = [rat1_rewards_dict[key] for key in rat1_rewards_dict.keys()]
matches_values = [rat1_matches_dict[key] for key in rat1_matches_dict.keys()]

# Create a single figure with dual y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot( rewards_values, marker='o', color='tab:blue', label='Rewards')
ax1.set_xlabel('Run')
ax1.set_ylabel('Rewards', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')


ax2 = ax1.twinx()

ax2.plot(matches_values, marker='x', color='tab:red', label='Matches')
ax2.set_ylabel('Matches', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')


plt.figure(figsize=(10, 6))
rat1_transition_values = [rat1_transitions_dict[key] for key in rat1_transitions_dict.keys()]
rat2_transition_values = [rat2_transitions_dict[key] for key in rat2_transitions_dict.keys()]
plt.plot(rat1_transition_values, marker='o', color='tab:blue')
plt.plot(rat2_transition_values, marker='x', color='tab:red')
plt.xlabel('Run')
plt.ylabel('# Transitions')

#%% Plot histograms for reward and match intervals

# Plot histogram for rat1 reward intervals in a grid layout
x_lim = (0, 700)  # Set the x-axis limits

fig, axes = plt.subplots(nrows=19, ncols=4, figsize=(20, 50))
for idx, (key, intervals) in enumerate(rat1_reward_intervals_dict.items()):
    row_idx = idx // 4
    col_idx = idx % 4
    ax = axes[row_idx, col_idx]
    ax.hist(intervals, bins=20, alpha=0.5)
    ax.set_title(f'{key}')
    ax.set_xlabel('Reward Interval')
    ax.set_ylabel('Frequency')
    ax.set_xlim(x_lim)  # Set x-axis limits
fig.tight_layout()
plt.show()


# Plot histogram for rat1 reward intervals in a grid layout (FIX THIS)
fig, axes = plt.subplots(nrows=19, ncols=4, figsize=(20, 50))
for idx, (key, intervals) in enumerate(rat1_match_intervals_dict.items()):
    row_idx = idx // 4
    col_idx = idx % 4
    ax = axes[row_idx, col_idx]
    ax.hist(intervals, bins=20, alpha=0.5)
    ax.set_title(f'{key}')
    ax.set_xlabel('Match Interval')
    ax.set_ylabel('Frequency')
    ax.set_xlim(x_lim)  # Set x-axis limits
fig.tight_layout()
plt.show()

#%% Plot mean inter-reward intervals for rats 1 and 2

# Plot rewards for rat1
plt.figure(figsize=(10, 6))
rat1_rewards_intervals = [rat1_mean_reward_interval_dict[key] for key in rat1_mean_reward_interval_dict.keys()]
rat2_rewards_intervals = [rat2_mean_reward_interval_dict[key] for key in rat2_mean_reward_interval_dict.keys()]

# Create a single figure with dual y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot( rat1_rewards_intervals, marker='o', color='tab:blue', label='Rewards')
ax1.set_xlabel('Run')
ax1.set_ylabel('Rat 1 inter-reward interval', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()

ax2.plot(rat2_rewards_intervals, marker='x', color='tab:red', label='Matches')
ax2.set_ylabel('Rat 2 inter-reward interval', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')


#%% Plot mean inter-match intervals for rats 1 and 2

# Plot rewards for rat1
plt.figure(figsize=(10, 6))
rat1_match_intervals = [rat1_mean_match_interval_dict[key] for key in rat1_mean_match_interval_dict.keys()]
rat2_match_intervals = [rat2_mean_match_interval_dict[key] for key in rat2_mean_match_interval_dict.keys()]


# Create a single figure with dual y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot( rat1_match_intervals, marker='o', color='tab:blue', label='Rewards')
ax1.set_xlabel('Run')
ax1.set_ylabel('Rat 1 inter-match interval', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()

ax2.plot(rat2_match_intervals, marker='x', color='tab:red', label='Matches')
ax2.set_ylabel('Rat 2 inter-match interval', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
