"""
Main script for food nutrient analysis.
This script runs the complete analysis pipeline from data processing to visualization.
"""

import os
import logging
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

from data_processing import download_data, load_data, preprocess_data, normalize_data
from clustering import find_optimal_clusters, cluster_data, reduce_dimensions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('food_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_analysis_pipeline():
    """Run the complete analysis pipeline."""
    try:
        logger.info("Starting food nutrient analysis pipeline")
        
        for dir_path in ['data/raw', 'data/processed', 'data/results', 'docs/images', 'docs/html']:
            Path(dir_path).mkdir(exist_ok=True, parents=True)
      
        logger.info("STEP 1: Data Processing")
        file_path = download_data()
        df = load_data(file_path)
        preprocessed_df, numeric_df, food_names, food_ids = preprocess_data(df)
        normalized_data, scaler = normalize_data(numeric_df)
       
        np.save('data/processed/normalized_data.npy', normalized_data)
        food_names.to_csv('data/processed/food_names.csv', index=False)
        nutrient_cols = numeric_df.columns.tolist()
        pd.DataFrame({'column': nutrient_cols}).to_csv('data/processed/nutrient_columns.csv', index=False)
      
        logger.info("STEP 2: Clustering Analysis")
        inertias, suggested_clusters = find_optimal_clusters(normalized_data)
        
        n_clusters = suggested_clusters
        
        kmeans, labels = cluster_data(normalized_data, n_clusters)
        
        embedding = reduce_dimensions(normalized_data)
        
        logger.info("STEP 3: Visualization")
        
        try:
            # Try to import the robust visualization module first
            from visualization import create_cluster_visualization, create_interactive_plot, create_cluster_summary_plot, export_cluster_data
            
            output_dir = Path('docs/images')
            output_dir.mkdir(exist_ok=True, parents=True)
            
            html_dir = Path('docs/html')
            html_dir.mkdir(exist_ok=True, parents=True)
            
            data_dir = Path('data/results')
            data_dir.mkdir(exist_ok=True, parents=True)
           
            create_cluster_visualization(
                embedding, labels, food_names, 
                save_path=output_dir / 'food_clusters.png'
            )
          
            create_interactive_plot(
                embedding, labels, food_names, 
                save_path=html_dir / 'clusters.html'
            )
          
            create_cluster_summary_plot(
                embedding, labels, food_names,
                save_path=output_dir / 'cluster_summary.png'
            )
            
            export_cluster_data(
                normalized_data, labels, food_names,
                save_path=data_dir / 'cluster_data.csv'
            )
            
            logger.info("Visualizations created successfully using robust_visualization")
            
        except ImportError:
            logger.info("robust_visualization module not found. Running visualization separately...")
            import subprocess
          
            viz_script = Path('src/robust_visualization.py')
            if viz_script.exists():
                logger.info(f"Running visualization script: {viz_script}")
                try:
                    subprocess.run(['python', str(viz_script)], check=True)
                    logger.info("Visualization script completed successfully")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error running visualization script: {e}")
            else:
                logger.warning("Visualization script not found. Visualizations not created.")
      
        results = {
            'total_foods': len(food_names),
            'n_clusters': n_clusters,
            'cluster_sizes': pd.Series(labels).value_counts().to_dict()
        }
        
        pd.DataFrame([results]).to_json('data/results/summary.json', orient='records')
        
        logger.info("Analysis pipeline completed successfully")
        return results
    
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {e}")
        traceback.print_exc()
        raise

def find_similar_foods(food_name, n_recommendations=5):
    """
    Find foods with similar nutrient profiles to the specified food.
    
    Args:
        food_name (str): Name of the food to find alternatives for
        n_recommendations (int): Number of recommendations to return
    
    Returns:
        dict: Information about similar foods
    """
    try:
        food_names = pd.read_csv('data/processed/food_names.csv').iloc[:, 0]
        labels = np.load('data/results/cluster_labels.npy')
        embedding = np.load('data/results/umap_embedding.npy')
       
        min_length = min(len(food_names), len(labels), embedding.shape[0])
        food_names = food_names[:min_length]
        labels = labels[:min_length]
        embedding = embedding[:min_length]
       
        try:
            food_idx = food_names.str.lower().eq(food_name.lower()).idxmax()
            if not food_names.iloc[food_idx].lower() == food_name.lower():
                matching_foods = food_names[food_names.str.lower().str.contains(food_name.lower())]
                if len(matching_foods) == 0:
                    return {"error": f"Food '{food_name}' not found in database"}
                elif len(matching_foods) > 1:
                    return {"error": f"Multiple matches found for '{food_name}'. Please be more specific.",
                            "matches": matching_foods.tolist()}
                food_idx = matching_foods.index[0]
        except:
            return {"error": f"Food '{food_name}' not found in database"}
      
        food_cluster = labels[food_idx]
      
        cluster_indices = np.where(labels == food_cluster)[0]
      
        query_point = embedding[food_idx]
        distances = np.sqrt(np.sum((embedding[cluster_indices] - query_point)**2, axis=1))
        
        sorted_indices = np.argsort(distances)
        similar_food_indices = [cluster_indices[i] for i in sorted_indices if cluster_indices[i] != food_idx]
       
        recommendations = food_names.iloc[similar_food_indices[:n_recommendations]].tolist()
        return {"query": food_names.iloc[food_idx], "cluster": int(food_cluster), 
                "recommendations": recommendations}
    
    except Exception as e:
        logger.error(f"Error finding similar foods: {e}")
        traceback.print_exc()
        return {"error": f"An error occurred: {str(e)}"}

if __name__ == "__main__":
    try:
        results = run_analysis_pipeline()
        print(f"Analysis completed. Identified {results['n_clusters']} food clusters.")
       
        test_food = "Lettmelk, 1,0 % fett, laktosefri"  
        similar = find_similar_foods(test_food)
        
        if "error" not in similar:
            print(f"\nFoods similar to {similar['query']} (Cluster {similar['cluster']}):")
            for i, food in enumerate(similar['recommendations'], 1):
                print(f"{i}. {food}")
        else:
            print(f"\nError: {similar['error']}")
    except Exception as e:
        print(f"Error in main: {e}")