from copy import deepcopy

from configuration.datasets import *
from shared.configuration_classes import ExperimentConfiguration
from shared.enums import ExperimentEnum

ml_temporal = ExperimentConfiguration('ml_temporal', dataset=ml_mr, experiment=ExperimentEnum.TEMPORAL,
                                        test_size=20, validation_size=50)
ml_1m_temporal = ExperimentConfiguration('ml_1m_temporal', dataset=ml_mr_1m,
                                           experiment=ExperimentEnum.TEMPORAL,
                                           test_size=20, validation_size=50)
yk_temporal = ExperimentConfiguration('yk_temporal', dataset=yelp,
                                        experiment=ExperimentEnum.TEMPORAL, test_size=20, validation_size=50)
ab_temporal = ExperimentConfiguration('ab_temporal', dataset=amazon_book,
                                        experiment=ExperimentEnum.TEMPORAL, test_size=20, validation_size=50)

experiments = [ml_temporal, ml_1m_temporal, yk_temporal, ab_temporal]
experiment_names = [e.name for e in experiments]

