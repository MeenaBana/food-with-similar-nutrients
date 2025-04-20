"""
Visualization module for food clustering results.
This module provides functions to visualize the clustering results using different plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging
import traceback


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_cluster_visualization(embedding, labels, food_names, save_path=None):
    """
    Create a scatter plot of the UMAP embedding colored by cluster.
    
    Args:
        embedding (numpy.ndarray): 2D embedding from UMAP
        labels (numpy.ndarray): Cluster labels
        food_names (pandas.Series): Names of food items
        save_path (str, optional): Path to save the plot
    """
    try:
        min_length = min(embedding.shape[0], len(labels))
        embedding = embedding[:min_length]
        labels = labels[:min_length]
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', 
                            alpha=0.7, s=20)
        plt.colorbar(scatter, label='Cluster')
        plt.title('Food Clusters (UMAP Projection)', fontsize=15)
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster visualization saved to {save_path}")
        
        plt.close()
    except Exception as e:
        logger.error(f"Error in create_cluster_visualization: {e}")
        traceback.print_exc()

def create_interactive_plot(embedding, labels, food_names, save_path=None):
    """
    Create an interactive scatter plot using plotly.
    
    Args:
        embedding (numpy.ndarray): 2D embedding from UMAP
        labels (numpy.ndarray): Cluster labels
        food_names (pandas.Series): Names of food items
        save_path (str, optional): Path to save the HTML file
    
    Returns:
        plotly.graph_objects.Figure: Interactive plotly figure
    """
    try:
        min_length = min(embedding.shape[0], len(labels), len(food_names))
        embedding = embedding[:min_length]
        labels = labels[:min_length]
        food_names = food_names[:min_length]
       
        labels_str = [f"Cluster {label}" for label in labels]
       
        df = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            'cluster': labels_str,
            'food_name': food_names
        })
       
        fig = px.scatter(
            df, x='x', y='y', color='cluster', hover_data=['food_name'],
            color_discrete_sequence=px.colors.qualitative.G10,
            labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
            title='Food Clustering Visualization'
        )
        
        fig.update_layout(
            width=1000,
            height=800,
            template='plotly_white',
            legend_title='Cluster'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    except Exception as e:
        logger.error(f"Error in create_interactive_plot: {e}")
        traceback.print_exc()

def create_cluster_summary_plot(embedding, labels, food_names, save_path=None):
    """
    Create a simple plot showing cluster summaries.
    
    Args:
        embedding (numpy.ndarray): 2D embedding from UMAP
        labels (numpy.ndarray): Cluster labels
        food_names (pandas.Series): Names of food items
        save_path (str, optional): Path to save the plot
    """
    try:
        min_length = min(embedding.shape[0], len(labels), len(food_names))
        embedding = embedding[:min_length]
        labels = labels[:min_length]
        food_names = food_names[:min_length]
       
        plt.figure(figsize=(16, 14))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', 
                            alpha=0.7, s=20)
       
        unique_clusters = np.unique(labels)
       
        table_data = []
        table_colors = []
        
        for cluster in unique_clusters:
            cluster_indices = np.where(labels == cluster)[0]
           
            center_x = np.mean(embedding[cluster_indices, 0])
            center_y = np.mean(embedding[cluster_indices, 1])
           
            plt.text(center_x, center_y, f"Cluster {cluster}\n({len(cluster_indices)} items)", 
                    fontsize=12, fontweight='bold', ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7))
            
            sample_foods = []
            for idx in cluster_indices[:5]: 
                if idx < len(food_names):
                    sample_foods.append(food_names[idx])
          
            color = plt.cm.viridis(cluster / max(unique_clusters))
            table_data.append([
                f"Cluster {cluster}",
                len(cluster_indices),
                "\n".join(sample_foods[:3])  # Show first 3 foods only
            ])
            table_colors.append([color, 'white', 'white'])
        
        plt.colorbar(scatter, label='Cluster')
        plt.title('Food Clusters Summary (UMAP Projection)', fontsize=15)
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.grid(alpha=0.3)
      
        if len(table_data) > 0:
            ax = plt.gca()
            ax_table = plt.axes([0.65, 0.05, 0.3, 0.9])
            ax_table.axis('off')
            
            the_table = ax_table.table(
                cellText=table_data,
                colLabels=["Cluster", "Size", "Sample Foods"],
                loc='center',
                cellColours=table_colors
            )
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(9)
            the_table.scale(1, 1.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster summary plot saved to {save_path}")
        
        plt.close()
    except Exception as e:
        logger.error(f"Error in create_cluster_summary_plot: {e}")
        traceback.print_exc()

def export_cluster_data(embedding, labels, food_names, save_path=None):
    """
    Export cluster data to CSV without trying to compute means.
    
    Args:
        embedding (numpy.ndarray): 2D embedding from UMAP
        labels (numpy.ndarray): Cluster labels
        food_names (pandas.Series): Names of food items
        save_path (str, optional): Path to save the CSV file
    """
    try:
        min_length = min(embedding.shape[0], len(labels), len(food_names))
        embedding = embedding[:min_length]
        labels = labels[:min_length]
        food_names = food_names[:min_length]
       
        df = pd.DataFrame({
            'cluster': labels,
            'food_name': food_names,
            'x': embedding[:, 0],
            'y': embedding[:, 1]
        })
      
        cluster_sizes = df.groupby('cluster').size().reset_index(name='count')
        
        cluster_foods = {}
        for cluster in df['cluster'].unique():
            cluster_foods[cluster] = df[df['cluster'] == cluster]['food_name'].tolist()
        
        summary_data = []
        for cluster, foods in cluster_foods.items():
            summary_data.append({
                'cluster': cluster,
                'size': len(foods),
                'sample_foods': ', '.join(foods[:5]) + ('...' if len(foods) > 5 else '')
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            
            summary_path = Path(save_path).parent / 'cluster_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            
            logger.info(f"Cluster data exported to {save_path}")
            logger.info(f"Cluster summary exported to {summary_path}")
        
        return df, summary_df
    except Exception as e:
        logger.error(f"Error in export_cluster_data: {e}")
        traceback.print_exc()

def main():
    """Main function to run the visualization pipeline."""

    try:
        embedding = np.load('data/results/umap_embedding.npy')
        labels = np.load('data/results/cluster_labels.npy')
        food_names = pd.read_csv('data/processed/food_names.csv', header=None).iloc[:, 0]
        
        logger.info(f"Data shapes: embedding={embedding.shape}, labels={len(labels)}, food_names={len(food_names)}")
        
        output_dir = Path('docs/images')
        output_dir.mkdir(exist_ok=True, parents=True)
        
        data_dir = Path('data/results')
        data_dir.mkdir(exist_ok=True, parents=True)
       
        create_cluster_visualization(
            embedding, labels, food_names, 
            save_path=output_dir / 'food_clusters.png'
        )
       
        html_dir = Path('docs/html')
        html_dir.mkdir(exist_ok=True, parents=True)
        create_interactive_plot(
            embedding, labels, food_names, 
            save_path=html_dir / 'clusters.html'
        )
       
        create_cluster_summary_plot(
            embedding, labels, food_names,
            save_path=output_dir / 'cluster_summary.png'
        )
       
        export_cluster_data(
            embedding, labels, food_names,
            save_path=data_dir / 'cluster_data.csv'
        )
        
        logger.info("Visualization pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()