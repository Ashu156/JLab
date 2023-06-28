# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:41:07 2023

@author: ashutoshshukla
"""

#%%

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import itertools
import matplotlib.pyplot as plt

#%%

def load_excel_file():
    filepath = filedialog.askopenfilename(title="Select Excel File", filetypes=(("Excel Files", "*.xlsx"), ("All Files", "*.*")))
    if filepath:
        try:
            excel_data = pd.read_excel(filepath, sheet_name=None)
            dataframes = {}
            for sheet_name, df in excel_data.items():
                dataframes[sheet_name] = df.dropna(axis=0)
            print("Dataframes loaded and nan rows dropped successfully!")
            return dataframes
        except Exception as e:
            print(f"Error loading Excel file: {e}")

root = tk.Tk()
root.withdraw()

dataframes = load_excel_file()

#%%


# Initialize an empty dictionary to store the split dataframes for each cohort
split_dataframes = {}

# Iterate over each cohort
for cohort, temp_df in dataframes.items():
    print(f"Cohort: {cohort}")
    
    # Specify the columns for which you want to find unique values
    columns_of_interest = ['Unnamed: 0', 'Unnamed: 1']
    
    # Find unique values in the specified columns and store them in a list
    rat_ids = []
    
    for column in columns_of_interest:
        rat_ids.extend(temp_df[column].unique())
    
    # Remove duplicates from the list of unique values
    rat_ids = sorted(list(set(rat_ids)))
    print(rat_ids)

    # Generate paired pairs
    paired_pairs = list(itertools.combinations(rat_ids, 2))
    
    # Initialize a counter for each pair in the current cohort
    pair_counts = {pair: 0 for pair in paired_pairs}

    # Iterate over the pairs to match in the current cohort
    for pair in paired_pairs:
        if (((temp_df['Unnamed: 0'] == pair[0]) & (temp_df['Unnamed: 1'] == pair[1])) |
            ((temp_df['Unnamed: 0'] == pair[1]) & (temp_df['Unnamed: 1'] == pair[0]))).any():
            count = len(temp_df[((temp_df['Unnamed: 0'] == pair[0]) & (temp_df['Unnamed: 1'] == pair[1])) |
                                ((temp_df['Unnamed: 0'] == pair[1]) & (temp_df['Unnamed: 1'] == pair[0]))])
            pair_counts[pair] += count
            split_dataframes.setdefault(cohort, {})[pair] = temp_df[((temp_df['Unnamed: 0'] == pair[0]) & (temp_df['Unnamed: 1'] == pair[1])) |
                                                                    ((temp_df['Unnamed: 0'] == pair[1]) & (temp_df['Unnamed: 1'] == pair[0]))]
            split_dataframes[cohort][pair] = split_dataframes[cohort][pair].reset_index(drop=True)

    # Print the counts for each pair in the current cohort
    for pair, count in pair_counts.items():
        print(f"Matches found for pair {pair}: {count}")


#%% Plot rewards obtained per minute for all the cohorts separately

# Find the global minimum and maximum values for the y-axis range
global_min = float('inf')
global_max = float('-inf')

# Iterate over each cohort in split_dataframes
for cohort, cohort_data in split_dataframes.items():
    print(f"Cohort: {cohort}")
    
    # Iterate over each pair in the current cohort
    for pair, df in cohort_data.items():
        # Extract the relevant column for plotting
        column = 'Unnamed: 2'
        
        # Check if column exists in the current dataframe
        if column in df.columns:
            values = df[column]
            
            # Update the global minimum and maximum values
            local_min = values.min()
            local_max = values.max()
            
            if local_min < global_min:
                global_min = local_min
            if local_max > global_max:
                global_max = local_max

# Iterate over each cohort in split_dataframes again
for cohort, cohort_data in split_dataframes.items():
    print(f"Cohort: {cohort}")
    
    # Create a new figure for each cohort
    plt.figure()
    
    # Iterate over each pair in the current cohort
    for pair, df in cohort_data.items():
        # Extract the relevant column for plotting
        column = 'Unnamed: 2'
        
        # Check if column exists in the current dataframe
        if column in df.columns:
            values = df[column]
            
            # Plot the values for the current pair
            plt.plot(values, label = pair, marker = 'o')
            plt.axhline(2.0, linestyle = '--', color = 'red')
    
    # Set plot title and labels
    plt.title(f"Rewards data for Cohort: {cohort}")
    plt.xlabel('Training session #')
    plt.ylabel('Rewards/minute')

    # Set the x-axis and y-axis limits
    plt.xlim(0, len(df[column]))
    plt.ylim(global_min- 0.5, global_max + 0.5)
    
    # Add legend
    plt.legend()
    
    # Show the plot
    plt.show()

#%% end of script
