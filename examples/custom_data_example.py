import numpy as np
from clustering_metrics import CosineClusteringIndex, RevisedClusteringIndex
from sklearn.preprocessing import StandardScaler

def evaluate_custom_clustering(X, labels):
    """
    Evaluate clustering results using CCI and RCI metrics.
    
    Args:
        X: Input data matrix
        labels: Cluster labels
    """
    # Standardize data
    X_scaled = StandardScaler().fit_transform(X)
    
    # Calculate metrics
    cci = CosineClusteringIndex()
    rci = RevisedClusteringIndex()
    
    cci_scores = cci.score(X_scaled, labels)
    rci_scores = rci.score(X_scaled, labels)
    
    print("\nClustering Evaluation Results:")
    print("-" * 40)
    print(f"CCI Score: {cci_scores['CCI']:.3f}")
    print(f"RCI Score: {rci_scores['RCI']:.3f}")
    
    return cci_scores, rci_scores

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    labels = np.random.randint(0, 3, 100)
    
    # Evaluate clustering
    cci_scores, rci_scores = evaluate_custom_clustering(X, labels) 