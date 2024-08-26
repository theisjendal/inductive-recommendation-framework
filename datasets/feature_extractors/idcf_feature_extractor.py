import numpy as np

from datasets.feature_extractors.feature_extraction_base import FeatureExtractionBase
from shared.configuration_classes import FeatureConfiguration, ExperimentConfiguration


class IDCFFeatureExtractor(FeatureExtractionBase):
	"""
	Extracts in and out degree based on relation types.
	"""

	def __init__(self, use_cuda=False, use_bool=True):
		"""
		:param use_cuda: Irrelevant for this method
		:param use_bool: If true use bool (i.e., has relation as ingoing or outgoing edge) otherwise use degree.
		"""
		super().__init__(use_cuda, 'simple')
		self.use_bool = use_bool

	def extract_features(self, data_path: str, feature_configuration: FeatureConfiguration,
						 experiment: ExperimentConfiguration, other_experiment: ExperimentConfiguration = None,
						 fold: str = 0) -> np.ndarray:
		ws_users, = self.load_experiment(data_path, other_experiment, limit=['users'])
		users, = self.load_experiment(data_path, experiment, limit=['users'])

		ws_users = [u.index for u in ws_users]
		users = [u.index for u in users]

		mappings = self._get_mappings(data_path, experiment, other_experiment)

		features = np.zeros(np.max(users) - np.min(users) + 1)
		mapped = [mappings['user'][idx] for idx in ws_users]
		features[mapped] = 1

		return features
