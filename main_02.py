# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:41:07 2024

@author: pfari
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:13:44 2024

@author: faria
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64  # Import the base64 module
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go


# Function to load CSV file and return a DataFrame


def load_data(file):
    df = pd.read_csv(file, header = None)
    return df

def wide_long_clean_all(df0):
    
    df = df0
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


def wide_long_clean_dup (df0):
    
    df = df0
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

def wide_long_clean_std (df0):
    
    df = df0
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


def report_blk (df_blk):
    
    # Add new column with normalised value
    
    df_blk['Normalised'] = df_blk['Value'] / df_blk['Detection_Limit']
    
    df_blk = df_blk [df_blk['Normalised'] > 0]
    
    elem_out_list = list(df_blk["Element"].unique())
    
    print (f'The following elements have values > detection limit. Please check : {elem_out_list}')
    
    return df_blk
    

    
NTGS_std_list = ['AS08JAW004','AS08JAW006', 'DW06LMG005']

st.title("NTGS - Whole-rock geochem QAQC")

tab_titles = ["Data cleaning",
              "Duplicates",
              "Blanks",
              "Report"]

tabs = st.tabs(tab_titles)


with tabs[0]:
    
    st.title("Upload CSV file")
    uploaded_file = st.file_uploader("Upload Intertek CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load data into DataFrame
        df0 = load_data(uploaded_file)
        
        new_batch = uploaded_file.name
        st.write(f'### Batch: {new_batch}')
        
        st.write(df0.head(10))
        
        st.title ("Data in long format")
        df = wide_long_clean_all(df0)
        
        st.write(df.head(10))
        
        df_d = wide_long_clean_dup(df0)
        du_list = list(df_d["Sample"].unique())
        
with tabs[1]:
    st.title ("Duplicates")
    
    
    if uploaded_file is not None:
        dup_list = st.multiselect('Select duplicates', du_list)
    
        if st.checkbox('Plot of selected duplicates and report table', key = 'dup_plots'):
            plots, report_df = duplicates_px(df, dup_list)  # Calling duplicates() to get plots
            st.header('Plots of Duplicate Analysis')
            st.subheader('You can zoom in to see results')
            
            for plot in plots:
                #st.pyplot(plot)  # Display each plot using st.pyplot()
                st.plotly_chart(plot)
            st.write ("### Report")
            st.dataframe(report_df)
            
            if st.checkbox('Export report as csv'):
                # Export report_df to CSV and provide download link
                csv = report_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # Base64 encoding for download link
                href = f'<a href="data:file/csv;base64,{b64}" download="Duplicates_report_{new_batch}">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

with tabs[2]:
    st.title ("Blanks")
    
    if st.checkbox('Show me the blanks and report'):
        
        df_blk = wide_long_clean_blk(df0)
        report_blk = report_blk (df_blk)
        st.dataframe (report_blk)

    