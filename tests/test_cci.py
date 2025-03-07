import unittest
import numpy as np
from clustering_metrics import CosineClusteringIndex

class TestCCI(unittest.TestCase):
    def setUp(self):
        self.cci = CosineClusteringIndex()
        
    def test_perfect_clustering(self):
        X = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        labels = np.array([0, 0, 1, 1])
        scores = self.cci.score(X, labels)
        self.assertGreater(scores['CCI'], 0.5)
        
    def test_poor_clustering(self):
        X = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        labels = np.array([0, 0, 1, 1])
        scores = self.cci.score(X, labels)
        self.assertLess(scores['CCI'], 0.5)

if __name__ == '__main__':
    unittest.main() 