# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:59:47 2024

@author: pfari
"""


##WIDE TO LONG FORMAT##

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv ("C:/Pablo_offline/py/NTGS_geochem/checked/656_0_2411228.csv", header=None)

# %% Capture all NTGS reference values for future analysis

def wide_to_long (df):
    
    # Extract metadata
    elements_row = df.iloc[0].dropna().values.tolist()[1:]
    units_row = df.iloc[1].dropna().values.tolist()[1:]
    detection_row = df.iloc[2].dropna().values.tolist()[1:]
    method_row = df.iloc[3].dropna().values.tolist()[1:]
    
        
    # Find the indices of the start and end values, if they excist
    sample_ini = df.index[df[0]== 'SAMPLE NUMBERS'][0]+1
    sample_end = df.index[df[0]== 'CHECKS'][0]-2
    
    # Extract samples
    df_s = df.loc[sample_ini:sample_end]
    
    # Add column names to df_s
    columns = ["Sample"] + elements_row
    df_s.columns = columns 
    
    # Convert to long format
    df_long = pd.melt(df_s, id_vars=["Sample"], var_name="Element", value_name="Value")
    
    # Add metadata to the long format
    df_long["Unit"] = df_long["Element"].map(dict(zip(elements_row, units_row)))
    df_long["Detection_Limit"] = df_long["Element"].map(dict(zip(elements_row, detection_row)))
    df_long["Method"] = df_long["Element"].map(dict(zip(elements_row, method_row)))
    
    return (df_long)


# Folder containing the CSV files
input_folder = "C:/Pablo_offline/py/NTGS_geochem/checked/"
output_file = "C:/Pablo_offline/py/NTGS_geochem/merged_long_format.csv"
output_file_std = "C:/Pablo_offline/py/NTGS_geochem/All_NTGS-standards.csv"

# Initialize an empty list to store dataframes
all_long_dfs = []

# Loop through all CSV files in the folder
for file_name in os.listdir(input_folder):
    if file_name.startswith('6'):
        file_path = os.path.join(input_folder, file_name)
        # Read CSV into a dataframe
        df = pd.read_csv(file_path, header=None)
        # Transform the dataframe to long format
        long_df = wide_to_long(df)
        # Add a column with batch name
        long_df.insert(0,'Batch',file_name)
        # Append the long dataframe to the list
        all_long_dfs.append(long_df)

# Concatenate all dataframes in the list
merged_df = pd.concat(all_long_dfs, ignore_index=True)

# Export the merged dataframe to a CSV file
merged_df.to_csv(output_file, index=False)

print(f"All files have been processed and merged. The output is saved to {output_file}.")

## No I want to see all elements analysed in each standard

NTGS_Standards = ['AS08JAW004','AS08JAW006', 'DW06LMG005']

all_NTGS_Standards = merged_df[merged_df['Sample'].str.contains(r'DW06LMG005|AS08JAW004|AS08JAW006', na=False)].dropna(axis=1, how='all')


# Convert specific columns to numeric, coercing errors to NaN
all_NTGS_Standards["Value"] = pd.to_numeric(all_NTGS_Standards["Value"], errors="coerce")
all_NTGS_Standards["Detection_Limit"] = pd.to_numeric(all_NTGS_Standards["Detection_Limit"], errors="coerce")
# Drop NaN 
all_NTGS_Standards = all_NTGS_Standards.dropna()


# Unify Sample names (eg _A, DUP)

for i in NTGS_Standards:
    mask = all_NTGS_Standards["Sample"].str.contains(i)
    all_NTGS_Standards.loc[mask, "Sample"] = i


# Export the clean df to a CSV file
all_NTGS_Standards.to_csv(output_file_std, index=False)

print(f"All NTGS Standard values are saved to {output_file_std}.")

# %% 

# Choose the elements

oxides = ['BaO', 'CaO', 'Cr2O3','FeO', 'Fe2O3', 'K2O', 'MgO', 'MnO', 'Na2O', 'P2O5', 'SiO2' ]
REE =    coeff_dict = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']


all_NTGS_Standards_o = all_NTGS_Standards[all_NTGS_Standards["Element"].isin(oxides)]
all_NTGS_Standards_ree = all_NTGS_Standards[all_NTGS_Standards["Element"].isin(REE)]

# Major OXIDES: Group by 'Sample' and calculate statistics for 'Value'
stats = all_NTGS_Standards_o.groupby(["Sample", "Element"])["Value"].agg(
    count="count",
    mean="mean",
    std="std",
    min="min",
    max="max"
)

# Reset the index for better readability (optional)
stats.reset_index(inplace=True)

# Display the statistics
print(stats)

# REE: Group by 'Sample' and calculate statistics for 'Value'
stats = all_NTGS_Standards_ree.groupby(["Sample", "Element"])["Value"].agg(
    count="count",
    mean="mean",
    std="std",
    min="min",
    max="max",
    sum="sum"
)

# Reset the index for better readability (optional)
stats.reset_index(inplace=True)

# Display the statistics
print(stats)
# %%
# Plots

## Box plots for major oxides
for sample in NTGS_Standards:
    # Select only sample i
    df = all_NTGS_Standards_o.loc[all_NTGS_Standards_o["Sample"] == sample]
    
    # Create the box plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Element', y='Value', data=df)
    plt.yscale('log')

    # Add title and labels
    plt.title(f'Box Plot for major oxides - Sample {sample}')
    plt.xlabel('Element')
    plt.ylabel('Value')

# Show the plot
plt.show()


## Box plots for REE
for sample in NTGS_Standards:
    # Select only sample i
    df = all_NTGS_Standards_ree.loc[all_NTGS_Standards_ree["Sample"] == sample]
    
    # Create the box plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Element', y='Value', data=df)
    plt.yscale('log')

    # Add title and labels
    plt.title(f'Box Plot for REE - Sample {sample}')
    plt.xlabel('Element')
    plt.ylabel('Value')

# Show the plot
plt.show()

# %% Function 

def wide_long_clean_blk (df, batch_name=None):
    # This function takes a raw Intertek csv file and returns it in clean and in long format
        
    # Extract metadata
    elements_row = df.iloc[0].dropna().values.tolist()[1:]
    units_row = df.iloc[1].dropna().values.tolist()[1:]
    detection_row = df.iloc[2].dropna().values.tolist()[1:]
    method_row = df.iloc[3].dropna().values.tolist()[1:]
    
    # Extract blanks
    df_b = df[df.iloc[:,0] == 'Control Blank']
    
    # Add column names to df_b
    columns = ["Sample"] + elements_row
    df_b.columns = columns 
    
    # Convert to long format
    df_long = pd.melt(df_b, id_vars=["Sample"], var_name="Element", value_name="Value")
    
    # Add metadata to the long format
    df_long["Unit"] = df_long["Element"].map(dict(zip(elements_row, units_row)))
    df_long["Detection_Limit"] = df_long["Element"].map(dict(zip(elements_row, detection_row)))
    df_long["Method"] = df_long["Element"].map(dict(zip(elements_row, method_row)))
    
    # If no batch_name is provided, use 'Unknown' as the default
    if batch_name is None:
        batch_name = 'Unknown'
    
    # Add a column with batch name
    df_long.insert(0,'Batch',batch_name)
    return (df_long)

wide_long_clean_blk(df, 656_0_2411228)


# %% 

def wide_long_clean_s (df, batch = None):
    
    # Extract metadata
    elements_row = df.iloc[0].dropna().values.tolist()[1:]
    units_row = df.iloc[1].dropna().values.tolist()[1:]
    detection_row = df.iloc[2].dropna().values.tolist()[1:]
    method_row = df.iloc[3].dropna().values.tolist()[1:]
        
    # Find the indices of the start and end values, if they excist
    sample_ini = df.index[df[0]== 'SAMPLE NUMBERS'][0]+1
    sample_end = df.index[df[0]== 'CHECKS'][0]-2
    
    # Extract samples
    df_s = df.loc[sample_ini:sample_end]
    
    # Add column names to df_s
    columns = ["Sample"] + elements_row
    df_s.columns = columns 
    
    # Convert to long format
    df_long = pd.melt(df_s, id_vars=["Sample"], var_name="Element", value_name="Value")
    
    # Add metadata to the long format
    df_long["Unit"] = df_long["Element"].map(dict(zip(elements_row, units_row)))
    df_long["Detection_Limit"] = df_long["Element"].map(dict(zip(elements_row, detection_row)))
    df_long["Method"] = df_long["Element"].map(dict(zip(elements_row, method_row)))
    
    # Add a column with batch name
    df_long.insert(0,'Batch',batch)
    
    
    return (df_long)

wide_long_clean_s(df, 656_0_2411228)

# %%

def wide_long_clean_std (df, batch = None):
    
    # Extract metadata
    elements_row = df.iloc[0].dropna().values.tolist()[1:]
    units_row = df.iloc[1].dropna().values.tolist()[1:]
    detection_row = df.iloc[2].dropna().values.tolist()[1:]
    method_row = df.iloc[3].dropna().values.tolist()[1:]
        
    # Find the indices of the start and end values, if they excist
    sample_ini = df.index[df[0]== 'SAMPLE NUMBERS'][0]+1
    sample_end = df.index[df[0]== 'CHECKS'][0]-2
    
    # Extract samples
    df_s = df.loc[sample_ini:sample_end]
    
    # Add column names to df_s
    columns = ["Sample"] + elements_row
    df_s.columns = columns 
    
    # Convert to long format
    df_long = pd.melt(df_s, id_vars=["Sample"], var_name="Element", value_name="Value")
    
    # Add metadata to the long format
    df_long["Unit"] = df_long["Element"].map(dict(zip(elements_row, units_row)))
    df_long["Detection_Limit"] = df_long["Element"].map(dict(zip(elements_row, detection_row)))
    df_long["Method"] = df_long["Element"].map(dict(zip(elements_row, method_row)))
    
    # Add a column with batch name
    df_long.insert(0,'Batch',batch)
    
    # Extract Standards

    df_long = df_long[df_long['Sample'].str.contains(r'DW06LMG005|AS08JAW004|AS08JAW006', na=False)].dropna(axis=1, how='all')
    
    return (df_long)

wide_long_clean_std(df, "newbatch")

