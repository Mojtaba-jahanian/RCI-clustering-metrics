from clustering_metrics import CosineClusteringIndex, RevisedClusteringIndex, ClusteringVisualizer
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    # Load and preprocess data
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Calculate metrics
    cci = CosineClusteringIndex()
    rci = RevisedClusteringIndex()
    
    cci_scores = cci.score(X, labels)
    rci_scores = rci.score(X, labels)
    
    # Visualize results
    results = {
        'K-Means': {**cci_scores, **rci_scores}
    }
    
    visualizer = ClusteringVisualizer(save_dir='results')
    df = visualizer.plot_metrics_comparison(results)
    
    print("\nClustering Results:")
    print(df.to_string(float_format=lambda x: '{:.3f}'.format(x)))

if __name__ == "__main__":
    main() 