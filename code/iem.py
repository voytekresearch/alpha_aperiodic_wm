"""Class that enables training and testing of inverted encoding model (IEM)."""

# Import necessary modules
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata
import params

class IEM():
    """IEM model that can be trained to estimate labels from data."""
    def __init__(
            self, n_channels=params.IEM_N_CHANNELS,
            basis_func=params.IEM_BASIS_FUNC,
            feat_space_edges=params.IEM_FEAT_SPACE_EDGES,
            feature_name=params.FEATURE_NAME):
        """Initialize IEM with all required attributes."""
        self.n_channels = n_channels
        self.basis_func = basis_func
        self.feat_space_edges = feat_space_edges
        self.feature_name = feature_name

        self.feat_space_range = (np.diff(self.feat_space_edges)[0] + 1).astype(
            int)
        self.theta = np.linspace(*self.feat_space_edges, self.feat_space_range)
        self.channel_centers = np.linspace(
            self.feat_space_edges[0], self.feat_space_edges[-1] + 1,
            self.n_channels + 1)[:-1].astype(int)
        self.basis_set = np.array([np.roll(self.basis_func(
            self.theta), channel_center - len(
                self.theta) // 2) for channel_center in self.channel_centers])

        self.design_matrix = None
        self.weights = None
        self.estimated_crfs = None
        self.estimated_ctfs = None
        self.channel_offsets = None
        self.mean_channel_offset = None
        return

    def _gen_design_matrix(self, labels):
        """Generate design matrix to serve as teacher in training IEM."""
        self.design_matrix = self.basis_set @ np.eye(len(self.theta))[labels].T
        return

    def fit_iem(self, train_data):
        """Use training data to estimate weights for IEM that predict labels
        from data."""
        inv = np.linalg.inv(self.design_matrix @ self.design_matrix.T)
        self.weights = train_data @ self.design_matrix.T @ inv
        return

    def train_model(self, train_data, train_labels):
        """Train IEM model to estimate training labels from training data."""
        self._gen_design_matrix(train_labels)
        self.fit_iem(train_data)
        return

    def estimate_crf(self, test_data, test_labels):
        """Estimate channel response function (CRF) from testing data."""
        inv = np.linalg.inv(self.weights.T @ self.weights)
        self.estimated_crfs = inv @ self.weights.T @ test_data
        test_labels_idx = rankdata(test_labels, method='dense') - 1
        self.estimated_ctfs = np.array([np.roll(
            crf, -label_idx) for crf, label_idx in zip(
                self.estimated_crfs.T, test_labels_idx)]).T
        self.channel_offsets = self.basis_set.T @ self.estimated_ctfs
        self.mean_channel_offset = np.roll(np.mean(
            self.channel_offsets, axis=1), self.feat_space_range // 2)
        return

    def plot_basis_set(self):
        """Plot basis set for IEM model."""
        basis_set_df = pd.DataFrame({
            'Angular Location (°)': np.tile(self.theta, self.n_channels),
            'Peak Response (°)': np.repeat(self.channel_centers, len(
                self.theta)),
            'Channel Activation': self.basis_set.flatten()})
        axes = sns.lineplot(
            x='Angular Location (°)', y='Channel Activation',
            hue='Peak Response (°)', data=basis_set_df)
        axes.legend(labels=self.channel_centers, bbox_to_anchor=(
            1.05, 1), title='Peak Response (°)')
