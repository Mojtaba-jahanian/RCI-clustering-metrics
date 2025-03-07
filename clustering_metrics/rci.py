import numpy as np
from typing import Dict
from .cci import CosineClusteringIndex

class RevisedClusteringIndex(CosineClusteringIndex):
    """
    Revised Clustering Index (RCI) implementation.
    
    RCI is a modification of CCI that provides a balanced measure of clustering quality,
    ranging from -1 (poor clustering) to 1 (excellent clustering).
    """
    
    def score(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate RCI score and related metrics.
        
        Args:
            X: Input data matrix of shape (n_samples, n_features)
            labels: Cluster labels of shape (n_samples,)
            
        Returns:
            Dictionary containing:
                - RCI: Revised Clustering Index
                - Cohesion: Internal cluster similarity
                - Separation: Between-cluster separation
        """
        metrics = super().score(X, labels)
        cohesion = metrics['Cohesion']
        separation = metrics['Separation']
        
        total = cohesion + separation
        if total < self.eps:
            return {'RCI': 0.0, 'Cohesion': 0.0, 'Separation': 0.0}
        
        rci = (separation - cohesion) / total
        
        return {
            'RCI': rci,
            'Cohesion': cohesion,
            'Separation': separation
        } 