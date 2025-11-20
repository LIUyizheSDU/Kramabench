from smolagents import tool
from smolagents.tools import Tool
import os
import pandas as pd
from typing import List, Optional, Union, Dict
import fnmatch

@tool
def write_file(path: str, content: str) -> str:
    """
    Writes the given content to a file at the specified path.
    Args:
        path (str): The file path where the content should be written.
        content (str): The content to write to the file.
    Returns:
        str: Confirmation message indicating the file has been written.
    """
    with open(path, "w") as f:
        f.write(content)
    return f"written to {path}"

@tool                                                                                                                                                    
def process_beach_data(file_path:str) -> pd.DataFrame:
    """
    Processes a beach datasheet CSV file to extract relevant data.
    Args:
        file_path (str): Path to the beach datasheet CSV file.
    Returns:
        pd.DataFrame: A DataFrame containing processed data with columns:
            - beach
            - date
            - one_day_rain
            - enterococcus
            - sampling_location
    """                                                                                                                                                           
    # Extract beach name from file path                                                                                                                                                        
    beach_name = file_path.split('/')[-1].replace('_datasheet.csv', '').replace('_', ' ').title()                                                                                              
    if beach_name == "Pleasure Bay And Castle Island Beach":                                                                                                                                   
        beach_name = "Pleasure Bay and Castle Island Beach"                                                                                                                                    
                                                                                                                                                                                                
    # Load the datasheet - skip the first 2 rows which contain metadata                                                                                                                        
    df = pd.read_csv(file_path, skiprows=2, header=0)                                                                                                                                          
                                                                                                                                                                                                
    # List to store processed records                                                                                                                                                          
    processed_records = []                                                                                                                                                                     
                                                                                                                                                                                                
    # Process each row                                                                                                                                                                         
    for _, row in df.iterrows():                                                                                                                                                               
        date = row['Date']                                                                                                                                                                     
                                                                                                                                                                                                
        # Convert one_day_rain to float, handle errors                                                                                                                                         
        try:                                                                                                                                                                                   
            one_day_rain = pd.to_numeric(row['1-Day Rain'], errors='coerce')                                                                                                                   
            if pd.isna(one_day_rain):                                                                                                                                                          
                one_day_rain = 0.0                                                                                                                                                             
        except:                                                                                                                                                                                
            one_day_rain = 0.0                                                                                                                                                                 
                                                                                                                                                                                                
        # Find all Enterococcus columns (could be multiple sampling locations)                                                                                                                 
        enterococcus_cols = [col for col in df.columns if col.startswith('Enterococcus')]                                                                                                      
        tag_cols = [col for col in df.columns if col.startswith('Tag')]                                                                                                                        
                                                                                                                                                                                                
        # Process each Enterococcus measurement                                                                                                                                                
        for i, e_col in enumerate(enterococcus_cols):                                                                                                                                          
            # Get corresponding tag column if available                                                                                                                                        
            tag_col = tag_cols[i] if i < len(tag_cols) else None                                                                                                                               
                                                                                                                                                                                                
            # Get enterococcus value                                                                                                                                                           
            enterococcus_str = row[e_col]                                                                                                                                                      
                                                                                                                                                                                                
            # Skip if value is missing                                                                                                                                                         
            if pd.isna(enterococcus_str):                                                                                                                                                      
                continue                                                                                                                                                                       
                                                                                                                                                                                                
            # Get tag value                                                                                                                                                                    
            tag = row[tag_col] if tag_col and pd.notna(row[tag_col]) else None                                                                                                                 
                                                                                                                                                                                                
            # Process enterococcus value based on tag                                                                                                                                          
            try:                                                                                                                                                                               
                enterococcus_str = str(enterococcus_str).strip()                                                                                                                               
                if tag == '<':                                                                                                                                                                 
                    # For values labeled '< x', replace with x/2                                                                                                                               
                    enterococcus_value = float(enterococcus_str)                                                                                                                         
                elif tag == '>':                                                                                                                                                               
                    # For values labeled '> x', replace with x                                                                                                                                 
                    enterococcus_value = float(enterococcus_str)                                                                                                                               
                else:                                                                                                                                                                          
                    # For plain numbers, use as is                                                                                                                                             
                    enterococcus_value = float(enterococcus_str)                                                                                                                               
                                                                                                                                                                                                
                # Add record to processed list                                                                                                                                                 
                processed_records.append({                                                                                                                                                     
                    'beach': beach_name,                                                                                                                                                       
                    'date': date,                                                                                                                                                              
                    'one_day_rain': one_day_rain,                                                                                                                                              
                    'enterococcus': enterococcus_value,                                                                                                                                        
                    'sampling_location': f"Location_{i+1}"  # Simple location identifier                                                                                                       
                })                                                                                                                                                                             
            except:                                                                                                                                                                            
                # Skip invalid values                                                                                                                                                          
                continue                                                                                                                                                                       
                                                                                                                                                                                                
    # Create dataframe from processed records                                                                                                                                                  
    result_df = pd.DataFrame(processed_records)                                                                                                                                                
    print(f"Processed {len(result_df)} samples from {beach_name}, {len(result_df[result_df['enterococcus'] > 104])} exceedances")                                                                                                                             
                                                                                                                                                                                                
    return result_df

@tool
def list_filepaths(dataset_directory:str) -> list[str]:
   """
   This tool lists all of the file paths for relevant files in the data directory.


   Args:
        dataset_directory (str): The path to the dataset directory.
  
   Returns:
        list[str]: A list of file paths for all files in the dataset directory.
   """
   #dataset_directory = f"/Users/eylai/Projects/KramaBench/data/{workload}/input/"
   filepaths = []
   for root, _, files in os.walk(dataset_directory):
       for file in files:
           if file.startswith("."):
               continue
           filepaths.append(os.path.join(root, file))
   return filepaths

@tool
def list_input_filepaths(dataset_directory:str, files:list[str]) -> list[str]:
    """
    This tool lists all of the file paths for given files in the data directory.
    Args:
          dataset_directory (str): The path to the dataset directory.
          files (list[str]): A list of file names to look for.
    Returns:
          list[str]: A list of file paths for files found in the dataset directory.
    """
    # Step 1: Get all file paths in the dataset directory
    filepaths = []
    for root, _, files in os.walk(dataset_directory):
        for file in files:
            if file.startswith("."):
               continue
            filepaths.append(os.path.join(root, file))

    # Step 2: Match given file names to actual file paths
    selected_filepaths = []
    for pattern in files:
        #print(self.dataset.keys())
        #assert f in self.dataset.keys(), f"File {f} is not in dataset!"
        # Relaxed the assertion to a warning
        matching = [
            f for f in filepaths
            if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(os.path.basename(f), os.path.basename(pattern))
        ]
        if len(matching) == 0:
            print(f"WARNING: File {pattern} is not in dataset!")
        else: # only extend if there are matches
            selected_filepaths.extend(matching)
    return selected_filepaths

@tool
def read_csv(
    filepath: str,
    columns: Optional[List[str]] = None,
    n_rows: Optional[int] = None,
    row_indices: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Reads a CSV file and returns a DataFrame.
    
    You can optionally select specific columns, a number of rows, or specific row indices.

    Args:
        filepath (str): The path to the CSV file.
        columns (List[str], optional): List of column names to return.
        n_rows (int, optional): Number of rows to return from the top.
        row_indices (List[int], optional): Specific row indices to return.

    Returns:
        pd.DataFrame: The selected portion of the CSV file.

    Raises:
        ValueError: If the file is not a CSV.
    """
    if not filepath.endswith(".csv"):
        raise ValueError(f"Unsupported file type: {filepath}. Only CSV files are supported.")

    df = pd.read_csv(filepath, encoding="ISO-8859-1")

    if columns is not None:
        df = df[columns]

    if row_indices is not None:
        df = df.iloc[row_indices]
    elif n_rows is not None:
        df = df.head(n_rows)

    return df

@tool
def get_csv_metadata(filepath: str) -> Dict[str, object]:
    """
    Returns metadata about a CSV file, including column names,
    number of rows, and number of columns.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        Dict[str, object]: Metadata dictionary with keys:
            - 'columns': List of column names
            - 'n_rows': Number of rows
            - 'n_columns': Number of columns
            - 'column_types': Data types of each column
    """
    df = pd.read_csv(filepath, encoding="ISO-8859-1")

    metadata = {
        "columns": df.columns.tolist(),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "column_types": df.dtypes.apply(lambda dt: dt.name).to_dict()
    }

    return metadata

@tool
def summarize_dataframe(file_path: str) -> Dict[str, object]:
    """Summarizes a CSV file by providing metadata and sample data.
    Args:
        file_path (str): Path to the CSV file.
        Returns:
        dict: A summary dictionary containing:
            - file name
            - columns
            - missing values per column
            - data types of each column
            - sample values from the first 3 rows
            - potential type issues (if any)
    """
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    summary = {
        "file": os.path.basename(file_path),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "sample_values": df.head(3).to_dict(orient="list"),
    }

    # Optional anomaly check: inconsistent types
    type_issues = {}
    for col in df.columns:
        values = df[col].dropna().astype(str)
        if values.nunique() > 0:
            inferred_types = values.map(lambda v: type(eval(v)) if v.isdigit() else str).value_counts()
            if len(inferred_types) > 1:
                type_issues[col] = inferred_types.to_dict()
    if type_issues:
        summary["potential_type_issues"] = type_issues

    return summary

class ExploreDataTool(Tool):
    name = "explore_data"
    description = "Summarize a CSV file: columns, missing values, data types, sample values, and anomalies."

    def __call__(self, file_path: str):
        try:
            df = pd.read_csv(file_path)
            return summarize_dataframe(df, file_path)
        except Exception as e:
            return {"error": str(e)}