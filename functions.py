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
import plotly.express as px
import plotly.graph_objs as go

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

## Now I want to see all elements analysed in each standard

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
REE = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']

all_elements = list(all_NTGS_Standards.Element.unique())
delete_elements = ['WTTOT', 'Moisture105', 'Total', 'LOI-1000', 'C-Acinsol', 'C-CO3', 'LOI', 'Cl', 'Mn','P','Al','Ca','Fe','K','Mg','Mn','Na','Si','Ti']
all_elements = [item for item in all_elements if item not in delete_elements]
all_elements = ['Au', 'Ag', 'Al2O3', 'As', 'Ba', 'Be', 'Bi', 'CaO', 'Cd', 'Ce', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'F', 'Fe2O3', 'Ga', 'Gd', 'Ge', 'Hf', 'Ho', 'In', 'K2O', 'La', 'Lu', 'MgO', 'MnO', 'Mo', 'Na2O', 'Nb', 'Nd', 'Ni', 'P2O5', 'Pb', 'Pd', 'Pr', 'Pt', 'Rb', 'Re', 'S', 'Sb', 'Sc', 'Se', 'SiO2', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Te', 'Th', 'TiO2', 'Tl', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr', 'BaO', 'C', 'Cr2O3', 'FeO', 'SO3', 'Li']


all_NTGS_Standards_o = all_NTGS_Standards[all_NTGS_Standards["Element"].isin(oxides)]
all_NTGS_Standards_ree = all_NTGS_Standards[all_NTGS_Standards["Element"].isin(REE)]
all_NTGS_Standards_all = all_NTGS_Standards[all_NTGS_Standards["Element"].isin(all_elements)]

# Major OXIDES: Group by 'Sample' and calculate statistics for 'Value'
stats = all_NTGS_Standards_all.groupby(["Sample", "Element", "Unit"])["Value"].agg(
    count="count",
    mean="mean",
    std="std",
    min="min",
    max="max"
)

# Reset the index for better readability (optional)
stats.reset_index(inplace=True)

# Reset the index for better readability (optional)
stats.reset_index(inplace=True)

# Display the statistics
print(stats)

# Save the stats as pickle

stats.to_pickle('stats.pkl')



# %%
# Plots

## Box plots for major oxides
for sample in NTGS_Standards:
    # Select only sample i
    df = all_NTGS_Standards_all.loc[all_NTGS_Standards_all["Sample"] == sample]
    
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

def wide_long_clean_blk (df0):
    # This function takes a raw Intertek csv file and returns it in clean and in long format
    df = df0
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
    
    
    # Add a column with batch name
    df_long.insert(0,'Type','blk')
    
    # Filter unecessary "Elements' and drop Nan's
    elements_out = ['LOI', 'SiO2', 'Total']
    
    df_long = df_long[~df_long["Element"].isin(elements_out)].dropna()
    
    # Transform all values to numeric and det limit
    
    df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
    df_long["Detection_Limit"] = pd.to_numeric(df_long["Detection_Limit"], errors="coerce")
    
    return (df_long)

df_blk = wide_long_clean_blk(df)

## Now I will report

def report_blk (df_blk):
    
    # Add new column with normalised value
    
    df_blk['Normalised'] = df_blk['Value'] / df_blk['Detection_Limit']
    
    df_blk = df_blk [df_blk['Normalised'] > 0]
    
    elem_out_list = list(df_blk["Element"].unique())
    
    print (f'The following elements have values > detection limit. Please check : {elem_out_list}')
    
    return df_blk
    

report_blk(df_blk)

# %% 

def wide_long_clean_s (df):
    
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
    df_long.insert(0,'Type','sample')
    
    
    return (df_long)

df_sample = wide_long_clean_s(df)

# %%

def wide_long_clean_std (df):
    
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
    df_long.insert(0,'Type','std')
    
    # Extract Standards

    df_long = df_long[df_long['Sample'].str.contains(r'DW06LMG005|AS08JAW004|AS08JAW006', na=False)].dropna(axis=1, how='all')
    
    return (df_long)

df_std = wide_long_clean_std(df)

# %% Duplicates
# Make this work with long format

def wide_long_clean_dup (df):
    
    # Extract metadata
    elements_row = df.iloc[0].dropna().values.tolist()[1:]
    units_row = df.iloc[1].dropna().values.tolist()[1:]
    detection_row = df.iloc[2].dropna().values.tolist()[1:]
    method_row = df.iloc[3].dropna().values.tolist()[1:]
        
    # Find the indices of the start and end values, if they excist
    
    sample_end = df.index[df[0]== 'CHECKS'][0]-2
    dup_ini = sample_end + 3
    dup_end = df.index[df[0]== 'STANDARDS'][0]-2
    
    # Extract duplicates
    df_d = df.loc[dup_ini:dup_end]
    
    # Add column names to df_d
    columns = ["Sample"] + elements_row
    df_d.columns = columns
    
    # Convert to long format
    df_long = pd.melt(df_d, id_vars=["Sample"], var_name="Element", value_name="Value")
    
    # Add metadata to the long format
    df_long["Unit"] = df_long["Element"].map(dict(zip(elements_row, units_row)))
    df_long["Detection_Limit"] = df_long["Element"].map(dict(zip(elements_row, detection_row)))
    df_long["Method"] = df_long["Element"].map(dict(zip(elements_row, method_row)))
    
    # Add a column with batch name
    df_long.insert(0,'Type','dup')
    
    return (df_long)

df_dup = wide_long_clean_dup(df)

## Duplicate list

def dup_list (df_d):
    duplicate_list = list(df_d["Sample"].unique())
    return duplicate_list
    
dup_list = dup_list(df_dup)


def duplicates_px(df_all, dup_list):
    """
    This function takes a clean df and a list of samples. 
    It returns a series of scatter plots for duplicates with error thresholds 
    and a report highlighting elements outside the threshold.
    """

    plots = []  # List to store Plotly figures
    report_data = []  # List to store report information

    # Filter df to include only samples in dup_list
    df_dupl = df_all[df_all['Sample'].isin(dup_list)]

    for sample in dup_list:
        # Filter data for the specific sample
        df_i = df_dupl[df_dupl['Sample'] == sample]

        # Separate original and duplicate types
        df_sample = df_i[df_i['Type'] == 'sample']
        df_duplicate = df_i[df_i['Type'] == 'dup']

        # Convert 'Value' to numeric and align indices
        x = pd.to_numeric(df_sample['Value'], errors='coerce').reset_index(drop=True)
        y = pd.to_numeric(df_duplicate['Value'], errors='coerce').reset_index(drop=True)
        element_list = df_sample['Element'].reset_index(drop=True)

        # Create a boolean mask for positive values
        mask = (x > 0) & (y > 0)

        # Apply the mask
        x = x[mask].values
        y = y[mask].values
        element_list = element_list[mask].values

        # Define thresholds
        error_threshold = 0.15
        upper_threshold = (1 + error_threshold) * x
        lower_threshold = (1 - error_threshold) * x

        # Create scatter plot
        fig = go.Figure()

        # Original vs duplicate scatter
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='blue'), name='Duplicate vs Original'))

        # 1:1 line
        fig.add_trace(go.Scatter(x=x, y=x, mode='lines', line=dict(color='gray', dash='dash'), name='1:1 line'))

        # ±15% threshold bands
        fig.add_trace(go.Scatter(x=x, y=upper_threshold, mode='lines', line=dict(color='lightgray', width=0), name='+15% Threshold'))
        fig.add_trace(go.Scatter(x=x, y=lower_threshold, mode='lines', line=dict(color='lightgray', width=0), fill='tonexty', showlegend=False))

        # Points outside the threshold
        outside_threshold = (y > upper_threshold) | (y < lower_threshold)
        fig.add_trace(go.Scatter(x=x[outside_threshold], y=y[outside_threshold], mode='markers', marker=dict(color='red'), name='Outside Threshold'))

        # Layout adjustments
        fig.update_layout(
            title=f'Scatter Plot of Duplicate vs Original for {sample}',
            xaxis=dict(title='Original Measurements', type='log'),
            yaxis=dict(title='Duplicate Measurements', type='log'),
            legend=dict(x=1.02, y=1, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode='closest'
        )

        # Add element labels as tick texts
        fig.update_xaxes(tickvals=x, ticktext=element_list)

        # Generate report for outliers
        elements_outside = element_list[outside_threshold]
        report_data.append((sample, list(elements_outside)))

        # Append figure to the list
        plots.append(fig)

    # Convert report data into a DataFrame
    report_df = pd.DataFrame(report_data, columns=['SampleID', 'Elements outside the ±15% threshold'])

    return plots, report_df




# %% Function to convert df0 into long format

def wide_long_clean_all(df):
    # Create an empty list to store the individual DataFrames
    df_list = []
    
    # Define the sample types you want to process
    sample_types = ['sample', 'std', 'dup']
    
    for sample_type in sample_types:
        # Extract metadata
        elements_row = df.iloc[0].dropna().values.tolist()[1:]
        units_row = df.iloc[1].dropna().values.tolist()[1:]
        detection_row = df.iloc[2].dropna().values.tolist()[1:]
        method_row = df.iloc[3].dropna().values.tolist()[1:]
        
        # Find the indices of the start and end values based on sample type
        if sample_type == 'sample':
            sample_ini = df.index[df[0] == 'SAMPLE NUMBERS'][0] + 1
            sample_end = df.index[df[0] == 'CHECKS'][0] - 2
            df_s = df.loc[sample_ini:sample_end]
        elif sample_type == 'std':
            sample_ini = df.index[df[0] == 'SAMPLE NUMBERS'][0] + 1
            sample_end = df.index[df[0] == 'CHECKS'][0] - 2
            df_s = df.loc[sample_ini:sample_end]
        elif sample_type == 'dup':
            sample_end = df.index[df[0] == 'CHECKS'][0] - 2
            dup_ini = sample_end + 3
            dup_end = df.index[df[0] == 'STANDARDS'][0] - 2
            df_s = df.loc[dup_ini:dup_end]
        
        # Add column names to df_s
        columns = ["Sample"] + elements_row
        df_s.columns = columns 
        
        # Convert to long format
        df_long = pd.melt(df_s, id_vars=["Sample"], var_name="Element", value_name="Value")
        
        # Add metadata to the long format
        df_long["Unit"] = df_long["Element"].map(dict(zip(elements_row, units_row)))
        df_long["Detection_Limit"] = df_long["Element"].map(dict(zip(elements_row, detection_row)))
        df_long["Method"] = df_long["Element"].map(dict(zip(elements_row, method_row)))
        
        # Add a column with the sample type
        df_long.insert(0, 'Type', sample_type)
        
        # For 'std' (standards), filter specific samples
        if sample_type == 'std':
            df_long = df_long[df_long['Sample'].str.contains(r'DW06LMG005|AS08JAW004|AS08JAW006', na=False)].dropna(axis=1, how='all')
        
        # Append the current DataFrame to the list
        df_list.append(df_long)
    
    # Concatenate all DataFrames in the list into one DataFrame
    df_all = pd.concat(df_list, ignore_index=True)
    
    return df_all

df_all = wide_long_clean_all(df)
