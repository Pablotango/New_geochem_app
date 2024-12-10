
import streamlit as st

# Set the page configuration for a wide layout as the very first command
st.set_page_config(layout="centered")

import pandas as pd
import base64  # Import the base64 module
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st


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
    
    # Preprocess the 'Value' column to handle cases with < or >
    df_all["Value"] = df_all["Value"].astype(str).str.extract(r'(\d+\.?\d*)')[0]
    
    # Convert 'Value' and 'Detection_Limit' columns to numeric
    df_all["Value"] = pd.to_numeric(df_all["Value"], errors='coerce')
    df_all["Detection_Limit"] = pd.to_numeric(df_all["Detection_Limit"], errors='coerce')
    
    return df_all

    

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



def plot_list(df_std, batch_std, element_groups):
    plots = []  # List to store Plotly figures
    
    for sample in batch_std:
        # Select only sample i
        df = df_std.loc[df_std["Sample"] == sample]
        # Only selected elements
        df = df[df["Element"].isin(element_groups)]
        
        # Check if the dataframe is empty
        if df.empty:
            print(f"No data available for sample {sample} with selected elements.")
            continue
        
        # Create a Plotly figure
        fig = go.Figure()
        
        # Main line for 'Mean'
        fig.add_trace(go.Scatter(
            x=df["Element"], 
            y=df["mean"], 
            mode='lines+markers', 
            name='Expected mean', 
            line=dict(color='red', width=2),
            marker=dict(symbol='circle')
        ))

        # Main line for 'Value'
        fig.add_trace(go.Scatter(
            x=df["Element"], 
            y=df["Value"], 
            mode='lines+markers', 
            name='Value', 
            line=dict(color='orange', width=2),
            marker=dict(symbol='circle')
        ))

        # Shaded region for min and max
        fig.add_trace(go.Scatter(
            x=df["Element"], 
            y=df["min"], 
            fill=None, 
            mode='lines', 
            line=dict(color='rgba(0,0,0,0)'), # Fully transparent line
            name='Expected Min value',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df["Element"], 
            y=df["max"], 
            fill='tonexty', 
            mode='lines', 
            line=dict(color='rgba(0,0,0,0)'),  # Fully transparent line
            name='Expected Max-Min range',
            fillcolor='rgba(0, 0, 255, 0.2)'
        ))

        # Shaded region for mean ± std (standard deviation)
        fig.add_trace(go.Scatter(
            x=df["Element"], 
            y=df["mean"] + df["std"], 
            fill=None, 
            mode='lines', 
            line=dict(color='rgba(0,0,0,0)'),  # Transparent line
            name='Mean + Std',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df["Element"], 
            y=df["mean"] - df["std"], 
            fill='tonexty', 
            mode='lines', 
            line=dict(color='rgba(0,0,0,0)'),  # Transparent line
            name='1 std',
            fillcolor='rgba(255, 0, 0, 0.2)'  # Shaded region color (light red)
        ))

        # Adding labels and title
        fig.update_layout(
            title=f"Sample {sample}",
            xaxis_title="Element",
            yaxis_title="Value",
            template="plotly",
            showlegend=True,
            xaxis=dict(tickangle=45),
            #yaxis=dict(type='log'),  # Set the y-axis to log scale if needed
            plot_bgcolor='white'
        )
        
        # Append the figure to the list
        plots.append(fig)

    # Return all the figures as a list if needed for later
    return plots

def NTGS_report(df_std):
    # Delete nan
    df_std = df_std.dropna()
    # Create a new column 'test' based on the condition Value >= min and Value <= max
    df_std['test'] = (df_std['Value'] >= df_std['min']) & (df_std['Value'] <= df_std['max']) | (df_std['max'] < df_std['Detection_Limit'])
    
    # Check if all values in 'test' are True
    if df_std['test'].all():
        st.write('### All values are within the expected range :)')
    else:
        st.write('These elements are outside the expected range, please check.')

        # Show a subset of the DataFrame with relevant columns only if any 'test' is False
        df_subset = df_std[df_std['test'] == False][['Sample', 'Element', 'Value', 'Unit', 'Detection_Limit', 'min', 'max', 'test']]
        st.write(df_subset)

def plot_average(df_all, sample_list, option):
    # Filter the dataframe for samples
    df = df_all[df_all['Type'] == 'sample']
    
    # Filter based on the selected samples
    df = df[df["Sample"].isin(sample_list)]
    
    # Filter based on elements in the Option
    df = df[df["Element"].isin(option["Element"])]
    
    # Get option into a dataframe
    option_df = pd.DataFrame(option)
    
    norm_values = []
    
    # Normalize each sample and element
    for sample in sample_list:
        sample_df = df[df['Sample'] == sample]
        for element in sample_df['Element']:
            norm = option_df[option_df['Element'] == element]
            if not norm.empty:
                norm_value = norm['Value'].iloc[0]
                sample_df.loc[sample_df['Element'] == element, 'Normalisation'] = sample_df['Value'] / norm_value
        norm_values.append(sample_df)
    
    # Concatenate all sample dataframes
    norm_df = pd.concat(norm_values)
    
    # Create a line plot using Plotly
    fig = px.line(norm_df, 
                  x='Element', 
                  y='Normalisation', 
                  color='Sample', 
                  title='Normalization of Elements by Sample',
                  labels={'Element': 'Element', 'Normalisation': 'Normalized Value'})
    # Add a horizontal line at y=1
    fig.add_shape(
        type="line",
        x0=norm_df['Element'].min(),  # Start of x-axis
        x1=norm_df['Element'].max(),  # End of x-axis
        y0=1,  # y-coordinate where the line starts
        y1=1,  # y-coordinate where the line ends
        line=dict(color="red", width=2, dash="solid"),  # Style of the line
        xref="x",
        yref="y"
    )
    # Show the plot in Streamlit
    st.plotly_chart(fig)
    

    
NTGS_std_list = ['AS08JAW004','AS08JAW006', 'DW06LMG005']

st.title("NTGS - Whole-rock geochem QAQC")

stats_dict = {'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211], 'Sample': ['AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW004', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'AS08JAW006', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005', 'DW06LMG005'], 'Element': ['Ag', 'Al2O3', 'As', 'Au', 'Ba', 'BaO', 'Be', 'Bi', 'Bi', 'C', 'CaO', 'Cd', 'Ce', 'Co', 'Cr', 'Cr2O3', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'F', 'Fe2O3', 'FeO', 'Ga', 'Gd', 'Ge', 'Hf', 'Ho', 'In', 'K2O', 'La', 'Li', 'Lu', 'MgO', 'MnO', 'Mo', 'Na2O', 'Nb', 'Nd', 'Ni', 'P2O5', 'Pb', 'Pd', 'Pr', 'Pt', 'Rb', 'Re', 'S', 'SO3', 'Sb', 'Sc', 'Se', 'SiO2', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Te', 'Th', 'TiO2', 'Tl', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr', 'Ag', 'Al2O3', 'As', 'Au', 'Ba', 'BaO', 'Be', 'Bi', 'Bi', 'C', 'CaO', 'Cd', 'Ce', 'Co', 'Cr', 'Cr2O3', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'F', 'Fe2O3', 'FeO', 'Ga', 'Gd', 'Ge', 'Hf', 'Ho', 'In', 'K2O', 'La', 'Li', 'Lu', 'MgO', 'MnO', 'Mo', 'Na2O', 'Nb', 'Nd', 'Ni', 'P2O5', 'Pb', 'Pd', 'Pr', 'Pt', 'Rb', 'Re', 'S', 'SO3', 'Sb', 'Sc', 'Se', 'SiO2', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Te', 'Th', 'TiO2', 'Tl', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr', 'Ag', 'Al2O3', 'As', 'Au', 'Ba', 'BaO', 'Be', 'Bi', 'C', 'CaO', 'Cd', 'Ce', 'Co', 'Cr', 'Cr2O3', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'F', 'Fe2O3', 'FeO', 'Ga', 'Gd', 'Ge', 'Hf', 'Ho', 'In', 'K2O', 'La', 'Li', 'Lu', 'MgO', 'MnO', 'Mo', 'Na2O', 'Nb', 'Nd', 'Ni', 'P2O5', 'Pb', 'Pd', 'Pr', 'Pt', 'Rb', 'Re', 'S', 'SO3', 'Sb', 'Sc', 'Se', 'SiO2', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Te', 'Th', 'TiO2', 'Tl', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr'], 'Unit': ['ppm', '%', 'ppm', 'ppb', 'ppm', '%', 'ppm', '%', 'ppm', '%', '%', 'ppm', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', '%', '%', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppm', 'ppm', '%', '%', 'ppm', '%', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppb', 'ppm', 'ppb', 'ppm', 'ppm', '%', '%', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppb', 'ppm', '%', 'ppm', '%', 'ppm', '%', '%', 'ppm', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', '%', '%', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppm', 'ppm', '%', '%', 'ppm', '%', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppb', 'ppm', 'ppb', 'ppm', 'ppm', '%', '%', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppb', 'ppm', '%', 'ppm', 'ppm', '%', '%', 'ppm', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', '%', '%', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppm', 'ppm', '%', '%', 'ppm', '%', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppb', 'ppm', 'ppb', 'ppm', 'ppm', '%', '%', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', '%', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm'], 'count': [13, 12, 13, 13, 13, 10, 13, 1, 12, 10, 12, 13, 13, 13, 12, 10, 13, 13, 13, 13, 13, 12, 12, 11, 13, 13, 13, 13, 13, 12, 12, 13, 10, 13, 12, 12, 13, 12, 13, 13, 13, 12, 13, 11, 13, 11, 13, 12, 12, 10, 13, 12, 12, 12, 13, 13, 13, 13, 13, 12, 13, 12, 13, 13, 13, 12, 13, 13, 13, 13, 13, 16, 15, 17, 19, 18, 13, 18, 1, 16, 12, 15, 18, 18, 18, 18, 13, 18, 18, 18, 18, 18, 17, 15, 16, 18, 18, 18, 18, 18, 13, 15, 18, 11, 18, 15, 15, 17, 15, 17, 18, 18, 15, 17, 17, 18, 17, 18, 12, 16, 13, 16, 18, 11, 15, 18, 15, 18, 16, 18, 12, 18, 15, 17, 17, 17, 18, 16, 18, 18, 18, 18, 22, 18, 22, 24, 25, 16, 25, 25, 18, 18, 24, 25, 25, 21, 16, 22, 25, 25, 25, 25, 20, 18, 19, 22, 25, 25, 25, 25, 20, 18, 25, 18, 25, 18, 18, 25, 18, 25, 25, 25, 18, 25, 20, 25, 17, 22, 16, 21, 16, 24, 20, 16, 18, 25, 25, 25, 25, 25, 16, 25, 18, 24, 24, 25, 20, 25, 25, 25, 24, 25], 'mean': [0, 4.889166666666667, 0.5923076923076922, 3.3076923076923075, 2431.0307692307692, 0.269, 1.5961538461538463, 0.1, 0.09083333333333334, 0.274, 10.121666666666666, 0.07384615384615384, 105.17461538461538, 93.65384615384616, 1084.9166666666667, 0.149, 1.5069230769230768, 36.63076923076923, 3.4407692307692312, 1.26, 2.4892307692307694, 1503.6666666666667, 12.491666666666667, 8.865454545454545, 7.292307692307692, 6.893846153846154, 1.2576923076923079, 2.756153846153846, 0.5553846153846154, 0.04416666666666667, 1.7391666666666667, 44.4476923076923, 6.709999999999999, 0.09192307692307693, 21.608333333333334, 0.19275, 0.24615384615384617, 0.5441666666666667, 4.310769230769231, 58.113846153846154, 420.94615384615383, 0.39358333333333334, 20.807692307692307, 1.5272727272727271, 13.95023076923077, 2.972727272727273, 67.47615384615385, 0, 0.08441666666666668, 0.197, 0.009230769230769228, 37.65833333333333, 0.9333333333333332, 45.37416666666667, 10.668461538461539, 0.9769230769230769, 506.1169230769231, 0.5984615384615385, 0.7746153846153846, 0, 10.13846153846154, 0.6633333333333334, 0.3930769230769231, 0.2030769230769231, 1.0746153846153847, 141.58333333333334, 0, 13.644615384615385, 0.91, 78.38461538461539, 97.3153846153846, 0, 16.86733333333333, 0.2647058823529412, 7.052631578947368, 209.90555555555557, 0.02769230769230769, 0.5561111111111111, 1.03, 0.0825, 0.08333333333333333, 10.889333333333333, 0.10222222222222223, 18.073888888888888, 53.16111111111111, 360.0, 0.060000000000000005, 1.3505555555555555, 120.03888888888888, 3.73, 2.4155555555555552, 0.9222222222222223, 184.41176470588235, 10.72, 6.989375, 14.988888888888889, 3.316111111111111, 1.176111111111111, 1.67, 0.7872222222222223, 0.05615384615384615, 0.4106666666666667, 8.572222222222223, 8.200000000000001, 0.3373888888888889, 9.09, 0.16313333333333332, 0.011764705882352941, 2.099333333333333, 1.3447058823529412, 9.943888888888889, 226.0111111111111, 0.10406666666666667, 2.2823529411764705, 14.299999999999999, 2.3640555555555554, 13.770588235294117, 18.05611111111111, 8.333333333333333e-05, 0.1078125, 0.26, 0, 35.166666666666664, 1.1727272727272728, 47.806666666666665, 2.5594444444444444, 0, 252.46833333333336, 0.708125, 0.5503888888888889, 0, 0.9755555555555555, 0.7426666666666667, 0.07529411764705883, 0.35647058823529415, 0, 226.11111111111111, 10.6125, 20.950555555555557, 2.2605555555555554, 71.11111111111111, 59.12222222222223, 0, 12.649999999999999, 0.6227272727272727, 3.2083333333333335, 361.728, 0.03625, 8.113199999999999, 0.23920000000000002, 0.032777777777777774, 0.5472222222222222, 0, 41.8468, 3.6439999999999997, 12.333333333333334, 0, 1.4268181818181818, 17.56, 1.4036000000000002, 1.03, 0.3132, 863.2, 1.6366666666666667, 0.9957894736842107, 19.863636363636363, 1.4372, 1.4464, 3.7648, 0.29919999999999997, 0.0018, 5.398888888888889, 21.723200000000002, 3.6222222222222222, 0.22332000000000002, 0.37722222222222224, 0.0195, 3.1, 3.401111111111111, 27.93, 12.8576, 7.228000000000001, 0.0064444444444444445, 6.996, 0.05499999999999998, 4.2171199999999995, 16.941176470588236, 300.4127272727273, 0, 0.04628571428571429, 0.118125, 0.2041666666666667, 0, 0, 75.31555555555556, 2.0372, 3.424, 149.6064, 2.6236, 0.23495999999999997, 0, 39.684400000000004, 0.0961111111111111, 1.1575, 0.19333333333333336, 20.6116, 0, 10.856, 10.106, 1.4072, 5.666666666666667, 83.556], 'std': [0.3003374170871223, 0.30350852416764884, 1.216183606246427, 2.0970064133034736, 173.69861534957326, 0.0056764621219754716, 0.15756154678614018, 0, 0.015642792899510292, 0.010749676997731392, 0.04018894767409979, 0.06551570569190027, 9.13452481466571, 5.86026384283953, 135.97423441315058, 0.0031622776601683785, 0.18705545592081388, 10.9043863204511, 0.6231300230618214, 0.32083225108042573, 0.27566469785053305, 240.76406154850721, 0.3175140178246492, 0.22205240986594296, 0.8311252798280357, 1.2954184553104737, 0.1943430749196656, 0.177460721547918, 0.12433927943436221, 0.005149286505444373, 0.08173775508331886, 4.624620982390798, 4.571031976844324, 0.0935186804320056, 0.8022109599366143, 0.007098335272186272, 0.38213301483337053, 0.03117642854737689, 2.482846019988827, 6.873129489130775, 140.60417617899677, 0.11471978376844506, 2.9344112238691498, 7.43735045686176, 1.4431376784542622, 4.0578543367378055, 9.905121182551252, 0.008866005899279597, 0.013694447005552245, 0.004830458915396484, 0.11793087936453417, 4.637879282864556, 2.0419835870292813, 0.14418790349800883, 1.2686137194712857, 0.6734907152958708, 29.597579682753167, 0.9887436991273839, 0.1652077169613748, 0.0, 1.3353579180245154, 0.010730867399773207, 0.2801602105755626, 0.07122553561102171, 0.164347973550806, 10.51802205164012, 1.559832338130263, 1.7550290376734288, 0.19157244060668013, 4.093459448487341, 12.936888224108479, 0.47004919955255753, 0.12050291440854176, 0.8908852174035864, 5.158312430873388, 26.730275239493604, 0.004385290096535145, 0.2678484841426516, 0, 0.1950555476439126, 0.013026778945578597, 0.07685484153042442, 0.053857716204562936, 0.6545149878200238, 4.2074358390674576, 62.766607832696586, 0.0, 0.16067769061521242, 9.864800101662054, 0.24002450855252758, 0.122340999920051, 0.07772920985935393, 52.84536264367173, 0.17976174708287035, 0.11630240754171892, 1.2494181652409204, 0.1773046195240698, 0.3834079211322167, 0.23268256589308156, 0.0549658242007282, 0.007448111101102006, 0.009611501047232546, 0.41144730997891943, 0.45166359162544883, 0.043449956188820936, 0.2091820806310687, 0.005462425764996276, 0.6253822360576534, 0.046670067903263696, 3.326330781895907, 0.6245670395787024, 18.818573081146653, 0.0038999389494611017, 3.8415874476116065, 1.459880132065643, 0.09884299625414407, 1.7290765981739908, 0.8422458694319387, 0.005124953806148062, 0.015808093918411967, 0.00816496580927727, 0.10880410531470461, 1.7415341445032282, 2.9600982784667496, 0.27858485926109355, 0.22794406491950198, 1.2579272444704233, 14.941486755694301, 1.2249801018792101, 0.05124696707110508, 0.0624257134277991, 0.3796317445076327, 0.02491891612715897, 0.32935766362456986, 0.05476554305293022, 0.2461721659613809, 9.755172919677563, 40.41850854909584, 0.7931357885538692, 0.295086671659107, 4.4706742337789676, 13.339968937063912, 0.4344753516240376, 0.13464550406421272, 1.001870544903177, 4.745516572116451, 307.6959328622983, 0.005000000000000001, 0.5430908456848325, 0.09022749026765624, 0.016016739609309327, 0.26425935468010914, 0.05208514489603134, 1.9214246450659076, 1.3108775686539151, 27.715218442821868, 0.005, 0.23841513142368895, 19.061370185097747, 0.1615518492620867, 0.12938959257477656, 0.05528109984434101, 115.78773682907878, 0.3123534950698534, 0.17554093506370796, 1.0878311557298876, 0.17600946944222443, 0.15085533909455556, 0.2513649405413039, 0.04689705036922188, 0.011232471724986964, 0.15242698884418243, 1.1062501224105394, 0.21297764465826693, 0.03631427634783139, 0.25453083843105984, 0.0038233031607267983, 0.5500000000000002, 0.07242756128703319, 1.6532644474896736, 0.48874226336587706, 5.679826875295878, 0.006147426264153367, 0.7441326046702516, 3.825190053757907, 0.23706210859885074, 66.82775488918817, 22.173547986351448, 0.007110731326663947, 0.010257401508875708, 0.009810708435174289, 0.16301284736424027, 6.1604767416337625, 0.6663019835880224, 0.9536260348685325, 0.17444005656194153, 0.9075057391921367, 7.525815503983605, 0.2515101058274466, 0.03923803426948567, 0.0, 2.1451244719129945, 0.018830166313622745, 0.8352570549394283, 0.05247497678328022, 1.3179329775574053, 10.773261246936178, 1.661595618675013, 0.44529016756866896, 0.12863125592172378, 11.933390008771577, 5.130308632691282], 'min': [0, 4.25, 0, 1.0, 2199.2, 0.26, 1.2, 0.1, 0.07, 0.26, 10.06, 0, 97.0, 82.6, 945.0, 0.14, 1.1, 26.0, 3.0, 0.9, 2.2, 1232.0, 11.74, 8.45, 5.4, 5.1, 0.85, 2.5, 0.4, 0.04, 1.63, 40.5, 5.0, 0, 21.02, 0.181, 0, 0.48, 3.1, 51.8, 326.9, 0.337, 15.0, 0, 12.5, 0, 57.8, 0, 0.07, 0.19, 0, 28.0, 0, 45.24, 9.3, 0, 459.3, 0, 0.6, 0, 6.42, 0.64, 0, 0.1, 0.74, 125.0, 0, 12.4, 0.7, 71.0, 69.4, 0, 16.59, 0, 3.0, 169.7, 0.02, 0.31, 1.03, 0, 0.06, 10.82, 0, 16.9, 47.1, 198.0, 0.06, 0.9, 108.0, 3.3, 2.1, 0.8, 108.0, 10.6, 6.8, 12.0, 3.0, 0, 1.1, 0.7, 0.04, 0.4, 7.4, 7.6, 0.27, 8.55, 0.16, 0, 2.05, 0, 8.4, 180.0, 0.099, 0, 12.0, 2.192, 10.0, 16.1, 0, 0.09, 0.25, 0, 32.0, 0, 47.53, 2.2, 0, 216.06, 0, 0.5, 0, 0.7, 0.72, 0, 0.3, 0, 209.0, 0, 19.1, 1.6, 62.0, 47.0, 0, 12.38, 0, 0, 266.7, 0.03, 6.87, 0.15, 0, 0.34, 0, 36.5, 1.7, 0, 0, 1.1, 8.0, 1.1, 0.7, 0.2, 649.0, 1.12, 0.86, 17.6, 0.9, 1.17, 3.3, 0.2, 0, 4.88, 18.9, 3.4, 0.18, 0.2, 0.01, 1.2, 3.28, 23.9, 11.7, 3.0, 0, 5.9, 0, 3.7, 0, 247.0, 0, 0.025, 0.11, 0, 0, 0, 72.0, 1.65, 2.0, 134.1, 2.2, 0.2, 0, 34.3, 0.08, 0, 0.1, 17.4, 0, 7.7, 9.1, 1.1, 0, 77.0], 'max': [0.2, 5.06, 3.0, 7.0, 2834.5, 0.28, 1.8, 0.1, 0.13, 0.29, 10.22, 0.2, 125.72, 100.9, 1456.0, 0.15, 1.7, 63.0, 4.89, 1.86, 3.03, 1971.0, 12.71, 9.2, 8.6, 9.59, 1.6, 3.1, 0.81, 0.05, 1.9, 55.23, 19.7, 0.2, 23.39, 0.2, 0.5, 0.57, 10.77, 74.15, 747.0, 0.642, 24.8, 6.8, 17.279, 6.5, 88.56, 0.025, 0.114, 0.2, 0.3, 41.0, 5.0, 45.73, 13.33, 1.9, 584.4, 3.15, 1.13, 0, 12.2, 0.68, 0.65, 0.3, 1.3, 166.0, 4.0, 17.53, 1.28, 87.0, 116.0, 0.13, 17.01, 2.0, 21.0, 302.4, 0.03, 1.1, 1.03, 0.69, 0.11, 11.12, 0.2, 19.5, 60.7, 425.0, 0.06, 1.55, 148.0, 4.2, 2.7, 1.07, 295.0, 11.31, 7.2, 17.8, 3.63, 1.7, 2.05, 0.87, 0.067, 0.43, 9.3, 9.4, 0.4, 9.24, 0.174, 1.2, 2.22, 6.96, 11.1, 255.0, 0.114, 8.6, 17.0, 2.5, 18.0, 19.9, 0.004, 0.14, 0.27, 0.15, 39.0, 8.0, 48.53, 3.0, 2.0, 269.8, 3.8, 0.66, 0.07, 2.14, 0.81, 0.3, 0.5, 0.2, 243.0, 162.0, 22.0, 2.9, 82.0, 110.0, 0.44, 13.08, 2.2, 15.0, 1836.0, 0.04, 9.25, 0.53, 0.08, 1.56, 0.14, 44.76, 9.0, 64.0, 0.01, 2.14, 105.2, 1.7, 1.4, 0.4, 1148.0, 2.49, 1.6, 22.0, 1.76, 1.8, 4.3, 0.4, 0.02, 5.52, 23.58, 4.0, 0.3, 1.25, 0.03, 4.0, 3.55, 30.9, 13.87, 31.5, 0.017, 9.2, 4.3, 4.55, 276.0, 346.0, 0.023, 0.08, 0.15, 0.6, 4.0, 0.6, 76.45, 2.31, 7.0, 167.0, 3.2, 0.31, 0, 43.4, 0.16, 3.9, 0.4, 23.4, 28.0, 16.0, 10.8, 1.6, 19.0, 97.0]}

stats = pd.DataFrame(stats_dict)
stats = stats[stats['max'] > 0]


tab_titles = ["Data entry",
              "NTGS Standards",
              "Duplicates",
              "Blanks",
              "Check_anomaly",
              "Compare against reference"
              ]

tabs = st.tabs(tab_titles)


with tabs[0]:
    
    st.title("Upload CSV file")
    uploaded_file = st.file_uploader("Upload Intertek CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load data into DataFrame
        df0 = load_data(uploaded_file)
        
        new_batch = uploaded_file.name
        st.write(f'### Batch: {new_batch}')
        
        st.write(df0)
        
        #st.title ("Data in long format")
        df = wide_long_clean_all(df0)
        
        #st.write(df.head(10))
        
        df_d = df[df['Type'] == 'dup']
        du_list = list(df_d["Sample"].unique())
        df_s = df[df['Type'] == 'sample']
        count = len(df_s['Sample'].unique())
        
        st.header(f'Number of samples, inlcuding NTGS standards: {count} ')
        

with tabs[1]:
    
    
    if uploaded_file is not None:
        # Filter standards and merge with stats
        df_std = df[df['Type'] == 'std']
        options = list(df_std['Sample'].unique())
        df_std = pd.merge(df_std, stats, on=['Sample', 'Element', 'Unit'], how='inner')
        batch_std = list(df_std['Sample'].unique())
        
        if not options:
            st.write("Seems like there are no NTGS standards in this batch :(")
        else:
            st.subheader("NTGS standards found:")
            st.write(options)
            
            # Define the actual element groups
            oxides = ['Al2O3','BaO', 'CaO', 'Cr2O3','FeO', 'Fe2O3', 'K2O', 'MgO', 'MnO', 'Na2O', 'P2O5', 'SiO2']
            REE = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
            #all_elements = ['Au', 'Ag', 'Al2O3', 'As', 'Ba', 'Be', 'Bi', 'CaO', 'Cd', 'Ce', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'F', 'Fe2O3', 'Ga', 'Gd', 'Ge', 'Hf', 'Ho', 'In', 'K2O', 'La', 'Lu', 'MgO', 'MnO', 'Mo', 'Na2O', 'Nb', 'Nd', 'Ni', 'P2O5', 'Pb', 'Pd', 'Pr', 'Pt', 'Rb', 'Re', 'S', 'Sb', 'Sc', 'Se', 'SiO2', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Te', 'Th', 'TiO2', 'Tl', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr', 'BaO', 'C', 'Cr2O3', 'FeO', 'SO3', 'Li']
            
            # Create a list of the group names that will be shown in the radio button
            element_groups_names = ["Oxides", "REE"]
            
            # Use st.radio to display the list of group names, not the full list
            selected_group_name = st.radio("Select an element group:", element_groups_names, index=1)  # Default selection
            
            # Map the selected group name to the actual list of elements
            if selected_group_name == "Oxides":
                selected_group = oxides
            elif selected_group_name == "REE":
                selected_group = REE

            # Check if element groups are selected
            if selected_group:
                # run report
                NTGS_report(df_std)
                
                plots = plot_list(df_std, batch_std, selected_group)
                
                for plot in plots:
                    st.plotly_chart(plot)  # Display each plot using Plotly
                
            else:
                st.write("No element group selected.")
            
            
            

with tabs[2]:
    
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

with tabs[3]:
    st.subheader ("Blanks")
    if uploaded_file is not None:
        
        if st.checkbox('Show me the blanks and report'):
            
            df_blk = wide_long_clean_blk(df0)
            report_blk = report_blk (df_blk)
            st.dataframe (report_blk)

with tabs[4]:
    st.subheader("Anomalies")
    
    if uploaded_file is not None:
        st.write ('The following elements returned values > 1000 (ppm or ppb)')
        
        df_high = df[(df['Type'] == 'sample') & (df['Value'] > 1000)]
        elements_high = df_high['Element'].unique().tolist()
        st.write (elements_high)
    
        st.write (df_high)
        if elements_high:
            image_url = "https://raw.githubusercontent.com/Pablotango/New_geochem_app/main/Capture.JPG"
        
            st.image(image_url, caption = "Image from Github", use_column_width=False)
        else:
            st.subheader ("There are no anomalous values")
    
with tabs[5]:
    
    st.subheader ("Compare against average abundance")
    if uploaded_file is not None:
        
        df_sample = df[df['Type'] == 'sample']
        
        if uploaded_file is not None:
            st.write ('Compare your results with the following standards:') 
            
            # Options for element groups
            df_upper_dict = {'Element': ['Na', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Cs', 'Ba', 'La', 'Ce', 'Nd', 'Sm', 'Eu', 'Tb', 'Yb', 'Lu', 'Hf', 'Pb', 'Th', 'U'], 'Value': [28200.0, 84700.0, 27400.0, 25000.0, 10.0, 3600.0, 60.0, 124.5, 600.0, 35000.0, 10.0, 20.0, 110.0, 350.0, 22.0, 240.0, 25.0, 3.7, 700.0, 30.0, 64.0, 26.0, 4.5, 0.88, 0.64, 2.2, 0.32, 5.8, 15.0, 10.5, 2.5], 'Unit': ['ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm']}
            df_upper_REE_dict = {'Element': ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y'], 'Value': [30.0, 64.0, 7.1, 26.0, 4.5, 0.88, 3.8, 0.64, 3.5, 0.8, 2.3, 0.33, 2.2, 0.32, 22.0], 'Unit': ['ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm', 'ppm']}
            
            # Create a list of the group names that will be shown in the radio button
            element_groups_names = ["Upper crustal", "Upper crustal REE"]
            
            # Use st.radio to display the list of group names
            option = st.radio("Select an element group:", element_groups_names, index=1)
            
            if option == 'Upper crustal':
                option = df_upper_dict
            elif option == 'Upper crustal REE':
                option = df_upper_REE_dict
            
            # Select samples
            sample_list = st.multiselect('Select Samples', options=df_sample['Sample'].unique())
            
            # Plotting
            if sample_list:
                plot_average(df_sample, sample_list, option)
        
        
        
    
    
    
    

        
        
