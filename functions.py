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

def wide_long_clean_blk (df):
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
    
    print (df_blk)
    

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


def duplicates_px(df, dup_list):
    ''' This function takes a clean df, and a list of samples. It returns a series of plots of duplicates'''
    
    plots = []  # List to store Plotly figures
    report_data = []  # List to store report information
    
    # Filter df to include only samples in dup_list
    df_dup = df[df['SampleID'].isin(dup_list)]
    
    # Iterate through the list of duplicates and plot
    for sample in dup_list:
        df_i = df_dup[df_dup['SampleID'] == sample].dropna(axis=1)
        
        x = df_i.iloc[[0], 2:]
        y = df_i.iloc[[1], 2:]
        
        original = x.values.flatten()
        duplicate = y.values.flatten()
        
        element_list = x.columns.tolist()
        
        # Create a Plotly figure
        fig = go.Figure()
        
        # Scatter plot of original vs duplicate measurements
        fig.add_trace(go.Scatter(x=original, y=duplicate, mode='markers', marker=dict(color='blue'), name='Duplicate vs Original'))
        
        # 1:1 line
        fig.add_trace(go.Scatter(x=original, y=original, mode='lines', line=dict(color='gray', dash='dash'), name='1:1 line'))
        
        # ±25% error threshold lines
        error_threshold = 0.15  # 15% error threshold
        upper_threshold = [(1 + error_threshold) * x for x in original]
        lower_threshold = [(1 - error_threshold) * x for x in original]
        
        fig.add_trace(go.Scatter(x=original, y=upper_threshold, mode='lines', line=dict(color='lightgray', width=0), fill='tonexty', showlegend=False))
        fig.add_trace(go.Scatter(x=original, y=lower_threshold, mode='lines', line=dict(color='lightgray', width=0), fill='tonexty', showlegend=False))
        
        # Highlight dots outside the threshold in red
        outside_threshold = (duplicate > upper_threshold) | (duplicate < lower_threshold)
        fig.add_trace(go.Scatter(x=original[outside_threshold], y=duplicate[outside_threshold], mode='markers', marker=dict(color='red'), name='Outside Threshold'))
        
        # Layout customization
        fig.update_layout(title=f'Scatter Plot of Duplicate vs Original Measurements for {sample}',
                          xaxis_title='Original Measurements',
                          yaxis_title='Duplicate Measurements',
                          xaxis_tickangle=-45,
                          showlegend=True,
                          legend=dict(x=1.02, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
                          margin=dict(l=0, r=0, t=50, b=0),
                          hovermode='closest')
        # Update x and y axes to logarithmic scale
        fig.update_xaxes(type='log')
        fig.update_yaxes(type='log')
        
        # Update x-axis tick labels
        fig.update_xaxes(tickvals=original, ticktext=element_list)

        # Create report
        elements_outside = [element_list[i] for i in range(len(element_list)) if outside_threshold[i]]
        report = f'Elements outside the ±15% error threshold: {elements_outside}'
        
        # Store report in report_data
        report_data.append((sample, elements_outside))
        
        # Add report as annotation
        fig.add_annotation(
            x=0.5,
            y=-0.25,
            text=report,
            showarrow=False,
            font=dict(size=10),
            align='center',
            xref='paper',
            yref='paper'
        )
        
        # Append the figure to the list
        plots.append(fig)
        
    # Convert report_data to a DataFrame
    report_df = pd.DataFrame(report_data, columns=['SampleID', 'Elements outside the ±15% error threshold'])
    
    return plots, report_df  # Return list of Plotly figures and report data frame
