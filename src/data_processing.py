"""
Data processing module for Norwegian food nutrient analysis.
This module handles loading, cleaning, and preprocessing the food dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_data(output_dir='data/raw'):
    """
    Download the Norwegian food data if not already present.
    
    Args:
        output_dir (str): Directory to save the downloaded file
    
    Returns:
        str: Path to the downloaded file
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "food.xlsx")
    
    if not os.path.exists(output_path):
        try:
            import gdown
            url = "https://drive.google.com/uc?id=1yFE0HJB-71BzpuiR633-JneF7EqNeqVM"
            gdown.download(url, output_path, quiet=False)
            logger.info(f"Data downloaded successfully to {output_path}")
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
    else:
        logger.info(f"Data file already exists at {output_path}")
    
    return output_path

def load_data(file_path):
    """
    Load the food data from Excel file.
    """
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    """
    Preprocess the food dataframe by removing irrelevant columns,
    handling missing values, and normalizing data.
    
    Args:
        df (pd.DataFrame): Raw food dataframe
    
    Returns:
        tuple: (preprocessed_df, numeric_df, food_names)
            - preprocessed_df: DataFrame with all relevant columns
            - numeric_df: DataFrame with only numeric columns (for clustering)
            - food_names: Series of food names
    """
   
    food_names = df['Matvare'].copy()
    food_ids = df['MatvareID'].copy()
    
    ref_cols = [x for x in df.columns.to_list() if 'Ref' in x]
    logger.info(f"Found {len(ref_cols)} reference columns to remove")
    
    cols_to_drop = ['MatvareID', 'Matvare'] + ref_cols
    preprocessed_df = df.copy()
    
    numeric_df = df.drop(columns=cols_to_drop)
    
    # Convert all columns to numeric, replacing non-numeric values with NaN
    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
    
    # Fill missing values with 0 (assuming absence of nutrient)
    numeric_df = numeric_df.fillna(0)
    
    logger.info(f"Preprocessed data shape: {numeric_df.shape}")
    
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    preprocessed_df.to_excel(os.path.join(processed_dir, 'food_preprocessed.xlsx'), index=False)
    
    return preprocessed_df, numeric_df, food_names, food_ids

def normalize_data(numeric_df):
    """
    Normalize the numeric data using Min-Max scaling.
    
    Args:
        numeric_df (pd.DataFrame): DataFrame with only numeric columns
    
    Returns:
        tuple: (normalized_data, scaler)
            - normalized_data: NumPy array of normalized values
            - scaler: Fitted MinMaxScaler object
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(numeric_df)
    logger.info("Data normalized using Min-Max scaling")
    
    return normalized_data, scaler

def main():
    """Main function to run the data processing pipeline."""
    
    file_path = download_data()
    df = load_data(file_path)
    
    preprocessed_df, numeric_df, food_names, food_ids = preprocess_data(df)
    normalized_data, scaler = normalize_data(numeric_df)
    
    np.save('data/processed/normalized_data.npy', normalized_data)
    food_names.to_csv('data/processed/food_names.csv', index=False)
    food_ids.to_csv('data/processed/food_ids.csv', index=False)
    
    logger.info("Data processing completed successfully")
    
    return preprocessed_df, numeric_df, normalized_data, food_names

if __name__ == "__main__":
    main()