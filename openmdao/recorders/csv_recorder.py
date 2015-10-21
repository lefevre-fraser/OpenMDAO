""" Class definition for CsvRecorder, a recorder that
saves the output into a csv file."""

import csv
import numpy
import sys

from six import string_types

from openmdao.recorders.base_recorder import BaseRecorder
from openmdao.util.record_util import format_iteration_coordinate

class CsvRecorder(BaseRecorder):

    def __init__(self, out=sys.stdout):
        super(CsvRecorder, self).__init__()

        self._wrote_header = False
        self._parallel = False

        if out != sys.stdout:
            # filename or file descriptor
            if isinstance(out, string_types):
                # filename was given
                out = open(out, 'w')
            self.out = out
        self.writer = csv.writer(out)

    def startup(self, group):
        super(CsvRecorder, self).startup(group)

    def record_iteration(self, params, unknowns, resids, metadata):
        if self._wrote_header is False:
            self.writer.writerow([param for param in params] + [unknown for unknown in unknowns])
            self._wrote_header = True

        def munge(val):
            if isinstance(val, numpy.ndarray):
                return ",".join(map(str, val))
            return str(val)
        self.writer.writerow([munge(value['val']) for value in params.values()] + [munge(value['val']) for value in unknowns.values()])

        if self.out:
            self.out.flush()

    def record_metadata(self, group):
        pass
        # TODO: what to do here?
        # self.writer.writerow([param.name for param in group.params] + [unknown.name for unknowns in group.unknowns])
