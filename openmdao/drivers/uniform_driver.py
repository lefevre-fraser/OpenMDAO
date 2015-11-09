"""
OpenMDAO design-of-experiments driver implementing the Uniform method.
"""

from openmdao.drivers.predeterminedruns_driver import PredeterminedRunsDriver
from six import moves, iteritems
import numpy as np


class UniformDriver(PredeterminedRunsDriver):
    """Design-of-experiments Driver implementing the Uniform method.
    """

    def __init__(self, num_samples=1):
        super(UniformDriver, self).__init__()
        self.num_samples = num_samples

    def _build_runlist(self):
        """Build a runlist based on a uniform distribution."""

        def sample_var(metadata):
            if metadata.get('type', 'double') == 'double':
                return np.random.uniform(metadata['low'], metadata['high'])
            elif metadata.get('type') == 'enum':
                return np.random.choice(metadata['items'])
            elif metadata.get('type') == 'int':
                return np.random.randint(metadata['low'], metadata['high'] + 1)

        for i in moves.xrange(self.num_samples):
            yield dict(((key, sample_var(metadata)) for key, metadata in iteritems(self.get_desvar_metadata())))
