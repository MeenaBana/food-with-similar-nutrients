"""
Clustering module for grouping similar food items.
This module implements K-means clustering and methods to determine optimal cluster count.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import umap
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_optimal_clusters(data, max_clusters=15, random_state=42):
    """
    Find the optimal number of clusters using the Elbow Method.
    
    Args:
        data (numpy.ndarray): Normalized data for clustering
        max_clusters (int): Maximum number of clusters to consider
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (inertias, suggested_clusters)
            - inertias: List of inertia values for each cluster count
            - suggested_clusters: Suggested optimal number of clusters
    """
    logger.info(f"Finding optimal number of clusters (max: {max_clusters})...")
   
    inertias = []
    K_range = range(1, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, 
                       n_init=10, random_state=random_state)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    # Determine the elbow point (where the rate of decrease sharply changes)
    # Using a simple heuristic to detect the elbow point
    inertia_diffs = np.diff(inertias)
    elbow_pts = np.diff(inertia_diffs)
    suggested_clusters = np.argmax(elbow_pts) + 2  # +2 because diff reduces length and also started from 1
    
    logger.info(f"Suggested optimal number of clusters: {suggested_clusters}")
    
    output_dir = Path('data/results')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    elbow_df = pd.DataFrame({
        'k': list(K_range),
        'inertia': inertias
    })
    elbow_df.to_csv(output_dir / 'elbow_curve_data.csv', index=False)
    
    return inertias, suggested_clusters

def plot_elbow_curve(K_range, inertias, suggested_clusters, save_path=None):
    """
    Plot the Elbow Method curve to visualize optimal cluster count.
    
    Args:
        K_range (range): Range of cluster counts evaluated
        inertias (list): Inertia values for each cluster count
        suggested_clusters (int): Suggested optimal number of clusters
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances (Inertia)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    
    plt.axvline(x=suggested_clusters, color='r', linestyle='--', 
                label=f'Suggested k = {suggested_clusters}')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Elbow curve plot saved to {save_path}")
    
    plt.close()

def cluster_data(data, n_clusters, random_state=42):
    """
    Perform K-means clustering on the data.
    
    Args:
        data (numpy.ndarray): Normalized data for clustering
        n_clusters (int): Number of clusters to create
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (kmeans, labels)
            - kmeans: Fitted KMeans model
            - labels: Cluster labels for each data point
    """
    logger.info(f"Clustering data into {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, 
                   n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(data)
    
    logger.info("Clustering completed")
    
    output_dir = Path('data/results')
    output_dir.mkdir(exist_ok=True, parents=True)
    np.save(output_dir / 'cluster_labels.npy', labels)
    
    return kmeans, labels

def reduce_dimensions(data, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Reduce dimensions using UMAP for visualization.
    
    Args:
        data (numpy.ndarray): High-dimensional data
        n_neighbors (int): UMAP parameter controlling local vs global structure
        min_dist (float): UMAP parameter controlling compactness of projection
        random_state (int): Random seed for reproducibility
    
    Returns:
        numpy.ndarray: Low-dimensional embedding (2D)
    """
    logger.info("Reducing dimensions with UMAP...")
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                        n_components=2, random_state=random_state)
    embedding = reducer.fit_transform(data)
    
    logger.info("Dimension reduction completed")
   
    output_dir = Path('data/results')
    output_dir.mkdir(exist_ok=True, parents=True)
    np.save(output_dir / 'umap_embedding.npy', embedding)
    
    return embedding

def main():
    """
    Main function to run the clustering pipeline.
    
    Args:
        data_path (str): Path to the normalized data
    
    Returns:
        tuple: (labels, embedding)
            - labels: Cluster labels for each data point
            - embedding: Low-dimensional embedding for visualization
    """

    data_path='data/processed/normalized_data.npy'
    
    try:
        data = np.load(data_path)
        logger.info(f"Loaded normalized data with shape {data.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    inertias, suggested_clusters = find_optimal_clusters(data)
   
    output_dir = Path('docs/images')
    output_dir.mkdir(exist_ok=True, parents=True)
    plot_elbow_curve(range(1, len(inertias) + 1), inertias, suggested_clusters, 
                    save_path=output_dir / 'elbow_curve.png')
    
    kmeans, labels = cluster_data(data, suggested_clusters)
    
    embedding = reduce_dimensions(data)
    
    logger.info("Clustering analysis completed successfully")
    
    return labels, embedding

if __name__ == "__main__":
    main()