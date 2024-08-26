from torch import nn


class Parallel(nn.Sequential):
	def __init__(self, aggregator=None, *modules, **kwargs):
		super(Parallel, self).__init__(*modules)
		self.aggregator = aggregator
		self.kwargs = kwargs

	def forward(self, input):
		results = []
		for module in self:
			results.append(module(input))

		if self.aggregator is not None:
			return self.aggregator(results, **self.kwargs)
		else:
			return results
