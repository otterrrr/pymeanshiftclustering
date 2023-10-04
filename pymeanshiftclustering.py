"""Mean shift clustering algorithm.

Mean shift clustering aims to discover *blobs* in a smooth density of
samples. It is a centroid based algorithm, which works by updating candidates
for centroids to be the mean of the points within a given region. These
candidates are then filtered in a post-processing stage to eliminate
near-duplicates to form the final set of centroids.

Seeding is performed using a binning technique for scalability.
"""

# Original Authors:
#          Conrad Lee <conradlee@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Martino Sorbaro <martino.sorbaro@ed.ac.uk>


"""
Modified sklearn.cluster.MeanShift class to adapt attraction mechanism

The original one is assigning labels using nearest center like that of K-means
In this version, substitute the labeling part to label based on attraction 
same as Mean Shift Segmentation does so this borrows connected component labeling
from scipy

it's also worth mentioning that here provides a revised predict method that
assigns label by attration to the mode or return -1 if given datapoint is orphan
"""

# Modified by:
#         Taesik Yoon <taesik.yoon.02@gmail.com>


from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.utils import check_random_state, gen_batches, check_array
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
from joblib import Parallel
from collections import Counter
import functools

class _FuncWrapper:
    """"Load the global configuration before calling the function."""
    def __init__(self, function):
        self.function = function
        functools.update_wrapper(self, self.function)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

def delayed(function):
    """Decorator used to capture the arguments of a function."""
    @functools.wraps(function)
    def delayed_function(*args, **kwargs):
        return _FuncWrapper(function), args, kwargs
    return delayed_function

def estimate_bandwidth(X, *, quantile=0.3, n_samples=None, random_state=0,
                       n_jobs=None):
    """Estimate the bandwidth to use with the mean-shift algorithm.

    That this function takes time at least quadratic in n_samples. For large
    datasets, it's wise to set that parameter to a small value.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input points.

    quantile : float, default=0.3
        should be between [0, 1]
        0.5 means that the median of all pairwise distances is used.

    n_samples : int, default=None
        The number of samples to use. If not given, all samples are used.

    random_state : int, RandomState instance, default=None
        The generator used to randomly select the samples from input points
        for bandwidth estimation. Use an int to make the randomness
        deterministic.
        See :term:`Glossary <random_state>`.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    bandwidth : float
        The bandwidth parameter.
    """
    X = check_array(X)

    random_state = check_random_state(random_state)
    if n_samples is not None:
        idx = random_state.permutation(X.shape[0])[:n_samples]
        X = X[idx]
    n_neighbors = int(X.shape[0] * quantile)
    if n_neighbors < 1:  # cannot fit NearestNeighbors with n_neighbors = 0
        n_neighbors = 1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                            n_jobs=n_jobs)
    nbrs.fit(X)

    bandwidth = 0.
    for batch in gen_batches(len(X), 500):
        d, _ = nbrs.kneighbors(X[batch, :], return_distance=True)
        bandwidth += np.max(d, axis=1).sum()

    return bandwidth / X.shape[0]

# separate function for each seed's iterative loop
def _mean_shift_single_seed(my_mean, X, nbrs, max_iter):
    # For each seed, climb gradient until convergence or max_iter
    bandwidth = nbrs.get_params()['radius']
    stop_thresh = 1e-3 * bandwidth  # when mean has converged
    completed_iterations = 0
    while True:
        # Find mean of points within bandwidth
        i_nbrs = nbrs.radius_neighbors([my_mean], bandwidth,
                                       return_distance=False)[0]
        points_within = X[i_nbrs]
        if len(points_within) == 0:
            break  # Depending on seeding strategy this condition may occur
        my_old_mean = my_mean  # save the old mean
        my_mean = np.mean(points_within, axis=0)
        # If converged or at max_iter, adds the cluster
        if (np.linalg.norm(my_mean - my_old_mean) < stop_thresh or
                completed_iterations == max_iter):
            break
        completed_iterations += 1
    return tuple(my_mean), len(points_within), completed_iterations

class MeanShift():
    """Mean shift clustering using a flat kernel.

    Mean shift clustering aims to discover "blobs" in a smooth density of
    samples. It is a centroid-based algorithm, which works by updating
    candidates for centroids to be the mean of the points within a given
    region. These candidates are then filtered in a post-processing stage to
    eliminate near-duplicates to form the final set of centroids.

    MODIFICATION ALERT:
    The binning technique from the original sklearn.cluster.MeanShift is
    removed. Since labeling by basin of attration requires all datapoints
    as seeds. This all the parameters related to binning & seeding are
    deprecated

    Read more in the :ref:`User Guide <mean_shift>`.

    Parameters
    ----------
    bandwidth : float, default=None
        Bandwidth used in the RBF kernel.

        If not given, the bandwidth is estimated using
        sklearn.cluster.estimate_bandwidth; see the documentation for that
        function for hints on scalability (see also the Notes, below).
    
    merge_bandwidth : float, default=None
        Bandwidth used in the phase of merging adjacent modes
        
        If not given, bandwidth*0.5

    n_jobs : int, default=None
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    max_iter : int, default=300
        Maximum number of iterations, per seed point before the clustering
        operation terminates (for that seed point), if has not converged yet.

        .. versionadded:: 0.22

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    n_iter_ : int
        Maximum number of iterations performed on each seed.

        .. versionadded:: 0.22

    Examples
    --------
    >>> from sklearn.cluster import MeanShift
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = MeanShift(bandwidth=2).fit(X)
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])
    >>> clustering.predict([[0, 0], [5, 5]])
    array([1, 0])
    >>> clustering
    MeanShift(bandwidth=2)

    Notes
    -----

    Scalability:

    Because this implementation uses a flat kernel and
    a Ball Tree to look up members of each kernel, the complexity will tend
    towards O(T*n*log(n)) in lower dimensions, with n the number of samples
    and T the number of points. In higher dimensions the complexity will
    tend towards O(T*n^2).

    Scalability can be boosted by using fewer seeds, for example by using
    a higher value of min_bin_freq in the get_bin_seeds function.

    Note that the estimate_bandwidth function is much less scalable than the
    mean shift algorithm and will be the bottleneck if it is used.

    References
    ----------

    Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
    feature space analysis". IEEE Transactions on Pattern Analysis and
    Machine Intelligence. 2002. pp. 603-619.

    """
    def __init__(self, *, bandwidth=None, merge_bandwidth=None, n_jobs=None, max_iter=300):
        self.bandwidth = bandwidth
        self.merge_bandwidth = merge_bandwidth
        self.n_jobs = n_jobs
        self.max_iter = max_iter
    
    def fit(self, X, y=None):
        """Perform clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to cluster.

        y : Ignored

        """
        self.X_ = X
        bandwidth = self.bandwidth
        if bandwidth is None:
            bandwidth = estimate_bandwidth(X, n_jobs=self.n_jobs)
        elif bandwidth <= 0:
            raise ValueError("bandwidth needs to be greater than zero or None,"
                             " got %f" % bandwidth)
        # computed or designated bandwidth for prediction
        self.bandwidth_ = bandwidth
        
        merge_bandwidth = self.merge_bandwidth
        if merge_bandwidth is None:
            merge_bandwidth = bandwidth*0.5
        elif merge_bandwidth > bandwidth:
            raise ValueError("bandwidth needs to be less than bandwidth,"
                             " got %f" % bandwidth)
        elif merge_bandwidth <= 0:
            raise ValueError("bandwidth needs to be greater than zero or None,"
                             " got %f" % bandwidth)
            
        seeds = X
        n_samples, n_features = X.shape

        # We use n_jobs=1 because this will be used in nested calls under
        # parallel calls to _mean_shift_single_seed so there is no need for
        # for further parallelism.
        nbrs = NearestNeighbors(radius=bandwidth, n_jobs=1).fit(X)
        self.nbrs_X = nbrs

        # execute iterations on all seeds in parallel
        all_res = Parallel(n_jobs=self.n_jobs)(
            delayed(_mean_shift_single_seed)
            (seed, X, nbrs, self.max_iter) for seed in seeds)
        self.n_iter_ = max([x[2] for x in all_res])

        # POST PROCESSING: finding connected components of modes
        # one connected component is one cluster of connected modes
        # to which attracts datapoints 
        modes = [tup[0] for tup in all_res]
        modes_n = len(modes)
        nbrs = NearestNeighbors(radius=merge_bandwidth, n_jobs=self.n_jobs).fit(modes)
        self.nbrs_modes = nbrs
        nbrs_coo = np.asarray([(row_idx, col_idx, 1) for row_idx,mode in enumerate(modes) for col_idx in nbrs.radius_neighbors([mode],return_distance=False)[0]])
        nbrs_coo_mat = coo_matrix((nbrs_coo[:,2],(nbrs_coo[:,0],nbrs_coo[:,1])),shape=(modes_n,modes_n))
        n_components, labels = connected_components(csgraph=nbrs_coo_mat, directed=False, return_labels=True)
        relabel_map = {cl[0]:i for i, cl in enumerate(Counter(labels).most_common())}
        labels = [relabel_map[label] for label in labels]
        
        self.labels_ = labels
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        seeds = X
        all_res = Parallel(n_jobs=self.n_jobs)(
            delayed(_mean_shift_single_seed)
            (seed, self.X_, self.nbrs_X, self.max_iter) for seed in seeds)
        preds = []
        for mode in [tup[0] for tup in all_res]:
            nbrs_indices = self.nbrs_modes.radius_neighbors([mode],return_distance=False)[0]
            preds.append(self.labels_[nbrs_indices[0]] if len(nbrs_indices) > 0 else -1)
        return preds
    
    def fit_predict(self, X, y=None):
        """
        Perform clustering on `X` and returns cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype=np.int64
            Cluster labels.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        self.fit(X)
        return self.labels_