import unittest
import numpy as np
from clustering_metrics import RevisedClusteringIndex

class TestRCI(unittest.TestCase):
    def setUp(self):
        self.rci = RevisedClusteringIndex()
        
    def test_perfect_clustering(self):
        X = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        labels = np.array([0, 0, 1, 1])
        scores = self.rci.score(X, labels)
        self.assertGreater(scores['RCI'], 0)
        
    def test_poor_clustering(self):
        X = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        labels = np.array([0, 0, 1, 1])
        scores = self.rci.score(X, labels)
        self.assertLess(scores['RCI'], 0)

if __name__ == '__main__':
    unittest.main() 