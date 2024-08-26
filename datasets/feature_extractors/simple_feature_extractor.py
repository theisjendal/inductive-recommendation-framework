
import numpy as np
import pandas as pd

from datasets.feature_extractors.feature_extraction_base import FeatureExtractionBase
from shared.configuration_classes import FeatureConfiguration, ExperimentConfiguration


class SimpleFeatureExtractor(FeatureExtractionBase):
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
						 experiment: ExperimentConfiguration, cold_experiment: ExperimentConfiguration=None,
						 fold: str = 0) -> np.ndarray:
		graph = self._load_graph(data_path, fold, feature_configuration, experiment)

		# Create dataframe with head, relation, and tail.
		heads, tails = graph.edges('uv')
		relations = graph.edata['type']
		unique_relations = np.unique(relations)
		triplet_df = pd.DataFrame.from_dict(dict(zip(['head', 'relation', 'tail'],
													 [heads.tolist(), relations.tolist(), tails.tolist()])))

		# Calculate in and out degree per relation and return as map to degree
		t_in = triplet_df.groupby(['head', 'relation']).count().reset_index().to_numpy()  # head, relation, count
		t_out = triplet_df.groupby(['tail', 'relation']).count().reset_index().to_numpy()  # tail, relation, count

		# We store in and out degree next to eachother, therefore multiply with 2 and plus 1 to out.
		t_in[:, 1] *= 2
		t_out[:, 1] = t_out[:, 1] * 2 + 1

		features = np.zeros((graph.num_nodes(), len(unique_relations)*2+1))
		for indices in [t_in, t_out]:
			if self.use_bool:
				features[indices[:, 0], indices[:, 1]] = 1
			else:
				features[indices[:, 0], indices[:, 1]] = indices[:, 2]

		features[:, -1] = graph.ndata['recommendable']

		return features
