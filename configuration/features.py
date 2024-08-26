from shared.configuration_classes import FeatureConfiguration
from shared.enums import FeatureEnum

# Base textual and degree extractor based on graphsage
graphsage = FeatureConfiguration('graphsage', 'graphsage', 'ckg', True, FeatureEnum.ENTITIES,
                                      attributes=['name', 'description'])

# Merging features
comsage = FeatureConfiguration('comsage', None, 'kg', True, FeatureEnum.ENTITIES, scale=True)
transsage = FeatureConfiguration('transsage', None, 'kg', True, FeatureEnum.ENTITIES, scale=True)

# KG based featrues
transr = FeatureConfiguration('transr_300', 'complex', 'kg', False, FeatureEnum.DESC_ENTITIES, model_name='transr',
                              embedding_dim=300)
complEX = FeatureConfiguration('complex', 'complex', 'kg', False, FeatureEnum.DESC_ENTITIES, model_name='complex')
transe = FeatureConfiguration('transe', 'complex', 'kg', False, FeatureEnum.DESC_ENTITIES, model_name='transe')


# Feature definitions
feature_configurations = [graphsage, comsage, transsage, transr, complEX, transe]
feature_conf_names = [f.name for f in feature_configurations]