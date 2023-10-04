# pymeanshiftclustering

Another version of sklearn.cluster.MeanShift with attraction labeling mechanism
Different from the original sklearn.cluster.MeanShift in that it assigns labels by "Basin of attraction" not by "Nearest neighbor"

### Installation

1. Download python package file: [pymeanshiftclustering-0.1.0.tar.gz](https://github.com/otterrrr/pymeanshiftclustering/blob/master/dist/pymeanshiftclustering-0.1.0.tar.gz)
1. pip install pymeanshiftclustering-0.1.0.tar.gz
    * Require three dependent modules: numpy, sklearn, scipy

### Example

Quite similar to sklearn.cluster.MeanShift
```python
# same example on https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
from pymeanshiftclustering import MeanShift
import numpy as np
X = np.array([[1, 1], [2, 1], [1, 0],
              [4, 7], [3, 5], [3, 6]])
clustering = MeanShift(bandwidth=2).fit(X)
clustering.labels_
clustering.predict([[0, 0], [5, 5]])
clustering
```

### Note

* MeanShift(bandwidth=None,<u>merge_bandwidth=None</u>,<s>seeds</s>,<s>bin_seeding</s>,<s>min_bin_freq</s>,<s>cluster_all</s>,n_jobs=None,max_iter=300)
  * a parameter added
    * <u>merge_bandwidth</u>: range of adjacent modes to merge, default=bandwidth*0.5
  * some parameters removed
    * <s>seeds</s>,<s>bin_seeding</s>,<s>min_bin_freq</s>,<s>cluster_all</s>
    * limited to seeding only by initial datapoints

### Future items(if they are in need)

* Internally merge modes within close range (<epsilon)

### Reference

* [sklearn.cluster.MeanShift - scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html)
* [sklearn.cluster.MeanShift - _mean_shift.py source code](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_mean_shift.py)
* [D. Comaniciu & P. Meer, 2002, Mean shift: a robust approach toward feature space analysis](https://courses.csail.mit.edu/6.869/handouts/PAMIMeanshift.pdf)