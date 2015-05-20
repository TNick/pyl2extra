"""
Code related to providing parameters following certain patterns.
"""
__authors__ = "Nicu Tofan"
__copyright__ = "Copyright 2015, Nicu Tofan"
__credits__ = ["Nicu Tofan"]
__license__ = "3-clause BSD"
__maintainer__ = "Nicu Tofan"
__email__ = "nicu.tofan@gmail.com"

from pylearn2.utils.rng import make_np_rng


class ParamStore(object):
    """
    Stores valid ranges for parameters and returns them according to plan.


    Following modes are defined:

    - rand_one: keep all parameters fixed except one that gets its value
      randomly from a list of possible values; on next iteration another
      parameter will get its value changed in the same way;
    - rand_all: change the values for all parameters on all passes, each
      according to its list of possible values;
    - seq_one: keep all parameters fixed except one that gets its value
      sequentially from its list of possible values
    - seq_all: change the values for all parameters on all passes; each
      parameter takes next value from its list of possible values.

    Assume two parameters (color and size) with following list of
    possible values:

    - color: white, red, green, blue, black
    - size: 1, 2, 3

    In this example:

    =========  =======  =======  =======  =======
      Mode        I       II       III      IV
    =========  =======  =======  =======  =======
    rand_one:  red/3    red/1    black/1  black/2
    rand_all:  white/3  red/1    blue/1   red/2
    seq_one:   white/1  white/2  red/2    green/2
    seq_all:   white/1  red/2    green/4  blue/1
    =========  =======  =======  =======  =======

    Parameters
    ----------
    parameters : list of lists
        Top level list contains one list for each parameter. Constituent
        lists are valid values for that particular parameter.
    mode : str, optional
        Indicates the way this instance is going to spit parameters.
    rng : RNG object or integer or list of integers, optional
        Used to generate random indices with ``rand_`` modes.
    """
    def __init__(self, parameters, mode=None, rng=None):
        if mode is None:
            mode = 'rand_one'
        #: the list of valid values for parameters
        self.parameters = parameters
        #: mode used to generate ext set of parameters
        self.mode = mode
        #: random number generator
        self.rng = make_np_rng(rng, which_method="random_integers")
        #: index of current parameter (used with ``_one`` variants)
        self.param_idx = None

        super(ParamStore, self).__init__()
        self.reset()

    def reset(self, mode=None):
        """
        Sets initial status.

        Parameters
        ----------
        mode : str, optional
            Indicates the way this instance is going to spit parameters.
            By default previous mode is used.
        """
        if not mode is None:
            self.mode = mode
        self.param_idx = 0
        self.crt_status = []
        # make sure we use first value in seq_all
        if self.mode == 'seq_all':
            init_crt = -1
        else:
            init_crt = 0
        for i in range(len(self.parameters)):
            self.crt_status.append(init_crt)
        # make sure we use first value in seq_one
        if self.mode == 'seq_one':
            self.crt_status[self.param_idx] = -1

    def next_seq_all(self):
        """
        Returns next value for all parameters.

        A parameter that reaches the end of its spectrum will be reset
        to first position.
        """
        result = []
        for i in range(len(self.parameters)):
            self.crt_status[i] = self.crt_status[i] + 1
            if self.crt_status[i] >= len(self.parameters[i]):
                self.crt_status[i] = 0
            result.append(self.parameters[i][self.crt_status[i]])
        return result

    def next_seq_one(self):
        """
        Returns next value for one parameter while keeping others fixed.

        A parameter that reaches the end of its spectrum will be reset
        to first position.
        """
        result = []
        for i in range(len(self.parameters)):
            if i == self.param_idx:
                self.crt_status[i] = self.crt_status[i] + 1
                if self.crt_status[i] >= len(self.parameters[i]):
                    self.crt_status[i] = 0
            result.append(self.parameters[i][self.crt_status[i]])

        # update the parameter that will be chenged next time
        self.param_idx = self.param_idx + 1
        if self.param_idx >= len(self.parameters):
            self.param_idx = 0
        return result

    def next_rnd_all(self):
        """
        Returns a random value fopr all parameters
        """
        result = []
        for i in range(len(self.parameters)):
            high = len(self.parameters[i])-1
            self.crt_status[i] = self.rng.random_integers(low=0,
                                                          high=high,
                                                          size=1)
            result.append(self.parameters[i][self.crt_status[i]])
        return result

    def next_rnd_one(self):
        """
        Returns a random value for one parameter while keeping others fixed.
        """
        result = []
        for i in range(len(self.parameters)):
            if i == self.param_idx:
                high = len(self.parameters[i])-1
                self.crt_status[i] = self.rng.random_integers(low=0,
                                                              high=high,
                                                              size=1)
            result.append(self.parameters[i][self.crt_status[i]])

        # update the parameter that will be chenged next time
        self.param_idx = self.param_idx + 1
        if self.param_idx >= len(self.parameters):
            self.param_idx = 0
        return result

    def next(self):
        """
        Retreive next set of parameters according to internal mode.
        """
        if self.mode == 'rand_one':
            return self.next_rnd_one()
        elif self.mode == 'rand_all':
            return self.next_rnd_all()
        elif self.mode == 'seq_one':
            return self.next_seq_one()
        elif self.mode == 'seq_all':
            return self.next_seq_all()
        else:
            raise AssertionError('%s is not among known modes for ParamStore'
                                 % self.mode)

    def __next__(self):
        """
        Interface for iterators.
        """
        self.next()

    def __iter__(self):
        """
        Interface for iterators.
        """
        return self
