"""Class that enables training and testing of inverted encoding model (IEM)."""

# Import necessary modules
from math import ceil
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
import params


class IEM:
    """IEM model that can be trained to estimate labels from data.

    Attributes
    ----------
    n_channels : int
        Number of channels to use in IEM.
    basis_func : function
        Basis function to use in IEM.
    feat_space_edges : tuple
        Tuple of (min, max) values for feature space.
    feature_name : str
        Name of feature to use in IEM.
    feat_space_range : int
        Number of values in feature space.
    theta : array
        Array of values in feature space.
    channel_centers : array
        Array of channel centers.
    channel_width : float
        Width of each channel.
    basis_set : array
        Array of basis functions for each channel.
    design_matrix : array
        Design matrix to serve as teacher in training IEM.
    weights : array
        Weights for IEM that predict labels from data.
    estimated_ctfs : array
        Estimated channel tuning functions (CTFs) from testing data.
    mean_channel_offset : array
        Mean channel offset from estimated channel tuning functions (CTFs).
    ctf_slope : array
        Slope of the channel tuning functions (CTF).

    Methods
    -------
    _gen_design_matrix(labels)
        Generate design matrix to serve as teacher in training IEM.
    fit_iem(train_data)
        Use training data to estimate weights for IEM that predict labels from
        data.
    train_model(train_data, train_labels)
        Train IEM model to estimate training labels from training data.
    estimate_ctf(test_data, test_labels)
        Estimate channel tuning functions (CTFs) from testing data.
    compute_mean_channel_offset()
        Compute mean channel offset from estimated channel tuning functions
        (CTFs).
    _avg_arr_across_equidistant_channels(arr, idx=0, dim=0)
        Take mean across channels that are equidistant from the tuned channel
        in channel tuning functions (CTFs).
    compute_ctf_slope()
        Compute slope of the channel tuning functions (CTF).
    plot_ctf()
        Plot channel tuning functions (CTF).
    plot_model_fit()
        Plot slope of the channel tuning functions (CTF).
    plot_mean_channel_offset()
        Plot mean channel offset from estimated channel tuning functions (CTF).
    plot_mean_channel_offset_slope()
        Plot slope of mean channel offset from estimated channel tuning
        functions (CTF).
    plot_mean_channel_offset_vs_ctf_slope()
        Plot mean channel offset vs. slope of the channel tuning functions
        (CTF).
    """

    def __init__(
        self,
        n_channels=params.IEM_N_CHANNELS,
        basis_func=params.IEM_BASIS_FUNC,
        feat_space_edges=params.IEM_FEAT_SPACE_EDGES,
        feature_name=params.FEATURE_NAME,
    ):
        """Initialize IEM with all required attributes.

        Parameters
        ----------
        n_channels : int (default: params.IEM_N_CHANNELS)
            Number of channels to use in IEM.
        basis_func : function (default: params.IEM_BASIS_FUNC)
            Basis function to use in IEM.
        feat_space_edges : tuple (default: params.IEM_FEAT_SPACE_EDGES)
            Tuple of (min, max) values for feature space.
        feature_name : str (default: params.FEATURE_NAME)
            Name of feature to use in IEM.
        """
        self.n_channels = n_channels
        self.basis_func = basis_func
        self.feat_space_edges = feat_space_edges
        self.feature_name = feature_name

        self.feat_space_range = (np.diff(self.feat_space_edges)[0] + 1).astype(
            int
        )
        self.theta = np.linspace(*self.feat_space_edges, self.feat_space_range)
        self.channel_centers = np.linspace(
            self.feat_space_edges[0],
            self.feat_space_edges[-1] + 1,
            self.n_channels + 1,
        )[:-1].astype(int)
        self.basis_set = np.array(
            [
                np.roll(
                    self.basis_func(self.theta),
                    channel_center - len(self.theta) // 2,
                )
                for channel_center in self.channel_centers
            ]
        )

        self.design_matrix = None
        self.weights = None
        self.inv_weights = None
        self.estimated_ctfs = None
        self.mean_channel_offset = None
        self.ctf_slope = None

    def _gen_design_matrix(self, labels):
        """Generate design matrix to serve as teacher in training IEM.

        Parameters
        ----------
        labels : array
            Array of labels to use in design matrix.
        """
        self.design_matrix = self.basis_set @ np.eye(len(self.theta))[labels].T

    def fit_iem(self, train_data):
        """Use training data to estimate weights for IEM that predict labels
        from data.

        Parameters
        ----------
        train_data : array
            Array of training data to use in IEM.
        """
        inv = np.linalg.inv(self.design_matrix @ self.design_matrix.T)
        self.weights = train_data @ self.design_matrix.T @ inv

    def train_model(self, train_data, train_labels):
        """Train IEM model to estimate training labels from training data.

        Parameters
        ----------
        train_data : array
            Array of training data to use in IEM.
        train_labels : array
            Array of training labels to use in IEM.
        """
        self._gen_design_matrix(train_labels)
        self.fit_iem(train_data)

    def estimate_ctf(self, test_data, test_labels):
        """Estimate channel tuning functions (CTFs) from testing data.

        Parameters
        ----------
        test_data : array
            Array of testing data to use in IEM.
        test_labels : array
            Array of testing labels to use in IEM.
        """
        inv = np.linalg.inv(self.weights.T @ self.weights)
        self.inv_weights = inv @ self.weights.T
        estimated_crfs = self.inv_weights @ test_data
        test_labels_idx = rankdata(test_labels, method="dense") - 1
        self.estimated_ctfs = np.array(
            [
                np.roll(crf, -label_idx)
                for crf, label_idx in zip(estimated_crfs.T, test_labels_idx)
            ]
        ).T

    def compute_mean_channel_offset(self):
        """Compute mean channel offset from estimated channel tuning functions
        (CTFs), which must already be computed.
        """
        assert self.estimated_ctfs is not None
        channel_offsets = self.basis_set.T @ self.estimated_ctfs
        self.mean_channel_offset = np.roll(
            np.mean(channel_offsets, axis=1), self.feat_space_range // 2
        )

    @staticmethod
    def _avg_arr_across_equidistant_channels(arr, idx=0, dim=0):
        """Take mean across channels that are equidistant from the tuned channel
        in channel tuning functions (CTFs), which must already be computed.
        """
        arr = np.moveaxis(arr, dim, 0)
        arr = np.roll(arr, -idx, axis=0)
        same_dist_to_tuned = np.array(
            list(
                zip(
                    arr.take(
                        indices=range(len(arr) // 2 + 1, len(arr)), axis=0
                    ),
                    arr.take(
                        indices=range(ceil(len(arr) / 2) - 1, 0, -1), axis=0
                    ),
                )
            )[::-1]
        )
        lst = [
            np.expand_dims(arr.take(indices=idx, axis=0), axis=0),
            np.mean(same_dist_to_tuned, axis=0 + 1),
            arr.take(indices=len(arr) // 2, axis=0)[not bool(len(arr) % 2)],
        ]
        arr = np.moveaxis(arr, 0, dim)
        return np.concatenate(lst)

    def compute_ctf_slope(self):
        """Compute the slope of the channel tuning functions (CTF), which must
        be already computed, in order to gauge stimulus selectivity."""
        assert self.estimated_ctfs is not None
        dist_from_tuned = np.min(
            [
                self.channel_centers,
                np.abs(self.channel_centers - self.feat_space_range),
            ],
            axis=0,
        )
        dist_from_tuned_avg = self._avg_arr_across_equidistant_channels(
            dist_from_tuned, idx=0, dim=0
        )
        ctf_avg_across_equidist_chs = (
            self._avg_arr_across_equidistant_channels(
                self.estimated_ctfs, idx=0, dim=0
            )
        )
        lin_model = LinearRegression()
        lin_model.fit(
            np.arange(1, len(dist_from_tuned_avg) + 1).reshape(-1, 1),
            np.mean(ctf_avg_across_equidist_chs, axis=1),
        )
        self.ctf_slope = -lin_model.coef_[0]

    def plot_basis_set(self):
        """Plot basis set for IEM model."""
        basis_set_df = pd.DataFrame(
            {
                "Angular Location (°)": np.tile(self.theta, self.n_channels),
                "Peak Response (°)": np.repeat(
                    self.channel_centers, len(self.theta)
                ),
                "Channel Activation": self.basis_set.flatten(),
            }
        )
        axes = sns.lineplot(
            x="Angular Location (°)",
            y="Channel Activation",
            hue="Peak Response (°)",
            data=basis_set_df,
        )
        axes.legend(
            labels=self.channel_centers,
            bbox_to_anchor=(1.05, 1),
            title="Peak Response (°)",
        )
