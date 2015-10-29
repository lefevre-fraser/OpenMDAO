""" Class definition for JsonRecorder, a recorder that
saves the output into a json file."""

import json
import sys

import numpy
from six import string_types

from openmdao.recorders.base_recorder import BaseRecorder


class JsonRecorder(BaseRecorder):
    def __init__(self, out=sys.stdout):
        super(JsonRecorder, self).__init__()

        self._first_entry = True
        self._parallel = False

        if out != sys.stdout:
            # filename or file descriptor
            if isinstance(out, string_types):
                # filename was given
                out = open(out, 'w')
        self.out = out

        self._results = dict()
        self._results['metadata'] = dict()
        self._results['iterations'] = []

    def startup(self, group):
        super(JsonRecorder, self).startup(group)

    def record_iteration(self, params, unknowns, resids, metadata):
        def munge(val):
            if isinstance(val, numpy.ndarray):
                return ",".join(map(str, val))
            return str(val)

        this_iteration = dict()
        for key, val in unknowns.iteritems():
            this_iteration[key] = munge(val)
        for key, val in params.iteritems():
            this_iteration[key] = munge(val)
        self._results['iterations'].append(this_iteration)

    def record_metadata(self, group):
        pass

    def close(self):
        json.dump(self._results, self.out, indent=4)
        super(JsonRecorder, self).close()
