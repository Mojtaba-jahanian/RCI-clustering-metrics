import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Optional
import os

class ClusteringVisualizer:
    """
    Visualization tools for clustering evaluation metrics.
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Args:
            save_dir: Directory to save visualizations. If None, current directory is used.
        """
        self.save_dir = save_dir or '.'
        os.makedirs(self.save_dir, exist_ok=True)
    
    def plot_metrics_comparison(self, results: Dict[str, Dict[str, float]], 
                              metrics: Optional[list] = None,
                              title: str = 'Clustering Metrics Comparison'):
        """
        Create comparison plots for different clustering methods.
        
        Args:
            results: Dictionary of clustering results
            metrics: List of metrics to plot
            title: Plot title
        """
        df = pd.DataFrame.from_dict(results, orient='index')
        
        if metrics is None:
            metrics = df.columns.tolist()
        
        # Bar plot
        plt.figure(figsize=(12, 6))
        df[metrics].plot(kind='bar', width=0.8)
        plt.title(title)
        plt.xlabel('Clustering Method')
        plt.ylabel('Score')
        plt.legend(title='Metric')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'metrics_comparison.png'), dpi=300)
        plt.close()
        
        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[metrics], annot=True, fmt='.3f', cmap='RdYlBu_r')
        plt.title('Clustering Metrics Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'metrics_heatmap.png'), dpi=300)
        plt.close()
        
        return df 