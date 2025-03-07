# Clustering Metrics Examples

This directory contains example scripts demonstrating the usage of CCI and RCI metrics.

## Files
- `iris_example.py`: Example using the Iris dataset
- `custom_data_example.py`: Example with custom data
# Cosine Clustering Index (CCI): A Balanced Approach to Evaluating Deep Clustering

This repository contains the implementation of algorithms presented in the following paper:

Jahanian, M., Karimi, A., Osati Eraghi, N., & Zarafshan, F. (2024). Introducing the Cosine Clustering Index (CCI): A Balanced Approach to Evaluating Deep Clustering. *SN Computer Science*, 5(687).
DOI: https://doi.org/10.1007/s42979-024-02970-7

## Files
- `iris_example.py`: Example using the Iris dataset
- `custom_data_example.py`: Example with custom data

## Usage
```bash
python iris_example.py
python custom_data_example.py
```

## Authors
- Mojtaba Jahanian
- Abbas Karimi
- Nafiseh Osati Eraghi
- Faraneh Zarafshan

Department of Computer, Arak Branch, Islamic Azad University, Arak, Iran
## Usage
```bash
python iris_example.py
python custom_data_example.py
```

```python:tests/test_cci.py
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
```

```python:tests/test_rci.py
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