import numpy as np
from typing import Dict, Union, Tuple
import warnings

class CosineClusteringIndex:
    """
    Cosine Clustering Index (CCI) implementation for evaluating clustering quality.
    
    The CCI metric evaluates clustering quality based on both cohesion (internal similarity)
    and separation (between-cluster distance) using cosine similarity.
    
    Attributes:
        eps (float): Small value to prevent division by zero
    """
    
    def __init__(self, eps: float = 1e-10):
        self.eps = eps
    
    def _cosine_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        
        if x_norm < self.eps or y_norm < self.eps:
            return 0.0
            
        return np.dot(x, y) / (x_norm * y_norm)
    
    def _calculate_cluster_centers(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate cluster centers as mean of points in each cluster."""
        unique_labels = np.unique(labels)
        centers = np.zeros((len(unique_labels), X.shape[1]))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            centers[i] = np.mean(X[mask], axis=0)
            
        return centers
    
    def score(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate CCI score and related metrics.
        
        Args:
            X: Input data matrix of shape (n_samples, n_features)
            labels: Cluster labels of shape (n_samples,)
            
        Returns:
            Dictionary containing:
                - CCI: Cosine Clustering Index
                - Cohesion: Internal cluster similarity
                - Separation: Between-cluster separation
        """
        if X.shape[0] != labels.shape[0]:
            raise ValueError("Number of samples in X and labels must be the same")
            
        centers = self._calculate_cluster_centers(X, labels)
        
        # Calculate cohesion
        cohesion = 0.0
        for i, center in enumerate(centers):
            mask = labels == i
            if np.any(mask):
                cluster_points = X[mask]
                for point in cluster_points:
                    cohesion += self._cosine_similarity(point, center)
        
        # Calculate separation
        separation = 0.0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                separation += self._cosine_similarity(centers[i], centers[j])
        
        # Calculate CCI
        total = cohesion + separation
        if total < self.eps:
            warnings.warn("Total (cohesion + separation) is close to zero")
            cci = 0.0
        else:
            cci = cohesion / total
        
        return {
            'CCI': cci,
            'Cohesion': cohesion,
            'Separation': separation
        } 