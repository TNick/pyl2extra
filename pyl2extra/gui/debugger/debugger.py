#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module implements a pylearn2 debugger.

The user needs to provide a signal handler that is used by the
debugger instance to send signals about its status.
Use `dbg_` class of methods to control the debugger.

Typical usage:

    debugger = Debugger(signal_handler)
    # at this point the debugger is in GROUND state
    debugger.load_file('/tmp/a/b/c')
    # debugger enters LOADING state, then STOPPED state
    debugger.dbg_run_one()
    # perform a step; enters RUNNING state, then PAUSED state
    debugger.dbg_stop()
    # enters STOPPED state
    debugger.unload_file()
    # back to GROUND state


@author: Nicu Tofan <nicu.tofan@gmail.com>
"""
import logging
import time
import thread

from pylearn2.utils import serial
from pylearn2.train import Train
from pylearn2.space import NullSpace
from pylearn2.utils.timing import log_timing

STS_LOADING = -2
STS_GROUND = -1
STS_STOPPED = 0
STS_RUNNING = 1
STS_PAUSED = 2

STS_REQ_STOP = 10
STS_REQ_PAUSE = 11

TRAIN_SECS_DOC = """\
The number of seconds that were spent in actual training during the most
recent epoch. This excludes seconds that were spent running callbacks for
the extensions, computing monitoring channels, etc."""
TOTAL_SECS_DOC = """\
The number of seconds that were spent on the entirety of processing for the
previous epoch. This includes not only training but also the computation of
the monitoring channels, running TrainExtension callbacks, etc. This value
is reported for the *previous* epoch because the amount of time spent on
monitoring for this epoch is not known until the monitoring channels have
already been reported."""

logger = logging.getLogger(__name__)

# set to true to use a worker thread
WITH_WORKER = True

class Runnable(object):
    """
    The worker thread for Debugger.
    """
    def __init__(self, debugger, loader=False):
        """
        Constructor
        """
        super(Runnable, self).__init__()
        self.debugger = debugger
        self.loader = loader

    def load(self):
        """
        If this is a load worker it will load the file then exit.
        """
        self.debugger._load_file()

    def train(self):
        """
        If this is a train worker it will loop-call the train method.
        """
        self.debugger._train_file()

    def run(self_ptr):
        """
        Build-in method.
        """
        if self_ptr.loader:
            self_ptr.load()
        else:
            self_ptr.train()

class Debugger(object):
    """
    Controlls the debugging.


    Parameters
    ----------

    signal_handler : obj
        An object that implements the signals mentioned below.

    Signals
    -------

    debug_end(yaml_file)
        Returning to ground state.
    debug_start(yaml_file)
        A file was succesfully loaded into the debugger.
    debug_run()
        Entering run state.
    debug_paused()
        Entering paused state.
    debug_stopped()
        Entering stopped state.
    debug_error(message)
        An error happened.

    Member Variables
    ----------------
    yaml_file : str
        The yaml file we're debugging or None if in ground state
    state : STS_...
        One of the status constants
    train_obj : pylearn2.Train
        The train object.
    worker : int
        Id of the timer used for running
    num_steps : int
        The number of steps to execute before entering paused state.
    crt_num_steps : int
        Number of steps performed since last run.
    """

    def __init__(self, signal_handler=None):
        """
        Constructor

        """
        super(Debugger, self).__init__()
        self.yaml_file = None
        self.tmp_fname = None
        self.signal_handler = signal_handler
        self.state = STS_GROUND
        self.train_obj = None
        self.worker = None
        self.num_steps = None
        self.crt_num_steps = None

    def is_running(self):
        """
        Tell if the debugger is in initialized state or not.
        """
        return self.yaml_file == None

    def _change_state(self, new_state):
        """
        The debugger changes its state.
        """
        if self.state != new_state:
            self.state = new_state
            self.signal_handler.debug_state_change(self.state, new_state)

    def unload_file(self):
        """
        Terminates debugging session.
        """

        if self.yaml_file is None:
            return

        # emit the signal and forget this file
        self._change_state(STS_GROUND)
        self.worker = None
        time.sleep(0.3)
        self.signal_handler.debug_end(self.yaml_file)
        if self.train_obj:
            self.train_obj.tear_down()
            self.train_obj = None
        self.yaml_file = None

    def load_file(self, fname):
        """
        Starts debugging given file.


        Parameters
        ----------
        fname : str
            The path to the file to load. Top level object must be a Train
            object.

        Returns
        -------
        init_ok : bool
            True if the file was succesfully loaded, False otherwise

        """
        if self.state != STS_GROUND:
            self.signal_handler.debug_error( "Can't load file if not in ground state")
            return
        self._change_state(STS_LOADING)
        self.tmp_fname = fname
        if WITH_WORKER:
            self.worker = Runnable(self, True)
            thread.start_new_thread(Runnable.run, (self.worker,))
        else:
            self._load_file()

    def dbg_run(self, num_steps=None):
        """
        Slot that instructs the debugger to run a number of steps.

        Parameters
        ----------
        num_steps : int
            Number of epochs to perform. Pass None to run without interuption.

        """
        if self.state == STS_GROUND:
            self.signal_handler.debug_error(
                      "Debugger is in ground state; no file to debug")
        elif self.state == STS_RUNNING:
            self.signal_handler.debug_error(
                      "Debugger is already running")
        else:
            if (self.state == STS_STOPPED) or (self.train_obj is None):
                self._load(self.yaml_file)
            self._change_state(STS_RUNNING)
            self.crt_num_steps = 0
            self.num_steps = num_steps
            self.signal_handler.debug_run()
            if WITH_WORKER:
                self.worker = Runnable(self)
                thread.start_new_thread(Runnable.run, (self.worker,))
            else:
                self._train_file()

    def dbg_stop(self):
        """
        Slot that instructs the debugger to terminate after current epoch.

        """
        if self.state == STS_GROUND:
            self.signal_handler.debug_error(
                      "Debugger is in ground state and cannot be stopped")
        elif self.state == STS_STOPPED:
            self.signal_handler.debug_error(
                      "Debugger is in stopped state already")
        else:
            self._change_state(STS_REQ_STOP)


    def dbg_pause(self):
        """
        Slot that instructs the debugger to pause after current epoch.

        """
        if self.state == STS_GROUND:
            self.signal_handler.debug_error(
                      "Debugger is in ground state and cannot be paused")
        elif self.state == STS_STOPPED:
            self.signal_handler.debug_error(
                      "Debugger is stopped and cannot be paused")
        else:
            self._change_state(STS_REQ_PAUSE)


    def dbg_continue(self):
        """
        Slot that instructs the debugger to run without interruption.

        """
        self.dbg_run(None)

    def dbg_run_one(self):
        """
        Slot that instructs the debugger to run one epoch then stop.

        """
        self.dbg_run(1)

    def perform_train(self):
        """
        Called to perform training.
        """

        # increment the number of steps that were taken
        self.crt_num_steps = self.crt_num_steps + 1

        # run the thing
        b_end = run_epoch(self.train_obj)

        if b_end:
            post_train(self.train_obj)
            self._change_state(STS_STOPPED)
            self.signal_handler.debug_stopped()
        else:
            # if the status is not running then exit
            if self.state != STS_RUNNING:
                if self.state == STS_REQ_PAUSE:
                    self._change_state(STS_PAUSED)
                    self.signal_handler.debug_paused()
                elif self.state == STS_REQ_STOP:
                    self._change_state(STS_STOPPED)
                    self.signal_handler.debug_stopped()
                    self.train_obj.tear_down()
                    self.train_obj = None
                self.signal_handler.debug_error(
                          "Invalid state %d while running" % self.state)
                return

            # see if we're done with this chunk
            if not self.num_steps is None:
                if self.crt_num_steps >= self.num_steps:
                    self._change_state(STS_PAUSED)
                    self.signal_handler.debug_paused()
                    self.crt_num_steps = 0
                    return



    def _load_file(self):
        """
        Loads the file async.
        """
        fname = self.tmp_fname
        try:
            self._load(fname)

            # set other states
            self._change_state(STS_STOPPED)
            self.yaml_file = fname
            self.signal_handler.debug_start(self.yaml_file)
        except Exception, exc:
            msg = 'Failed to load file in debugger'
            logger.error(msg, exc_info=True)
            self.signal_handler.debug_error(
                      '%s: %s' % (msg, str(exc)))
            self._change_state(STS_GROUND)
            self.yaml_file = None
        self.worker = None
        self.tmp_fname = None

    def _load(self, fname):
        """
        Internal load.
        """
        # load the train object
        train_obj = serial.load_train_file(fname)
        if not isinstance(train_obj, Train):
            raise ValueError('Top level object must be a pylearn2.Train')

        # prepare for training
        pre_train(train_obj)
        self.train_obj = train_obj
        self.crt_num_steps = 0

    def _train_file(self):
        """
        Train loop.
        """
        while self.state != STS_GROUND:
            if self.state == STS_STOPPED:
                time.sleep(0.5)
            elif self.state == STS_PAUSED:
                time.sleep(0.3)
            else:
                self.perform_train()
                time.sleep(0.1)

def pre_train(trainobj):
    """
    Code that is executed as the training starts.
    """
    trainobj.setup()
    if not trainobj.algorithm is None:
        if not hasattr(trainobj.model, 'monitor'):
            raise RuntimeError("The algorithm is responsible for setting"
                               " up the Monitor, but failed to.")
        if len(trainobj.model.monitor._datasets) > 0:
            # This monitoring channel keeps track of a shared variable,
            # which does not need inputs nor data.
            trainobj.training_seconds.__doc__ = TRAIN_SECS_DOC
            trainobj.model.monitor.add_channel(
                name="training_seconds_this_epoch",
                ipt=None,
                val=trainobj.training_seconds,
                data_specs=(NullSpace(), ''),
                dataset=trainobj.model.monitor._datasets[0])
            trainobj.total_seconds.__doc__ = TOTAL_SECS_DOC
            trainobj.model.monitor.add_channel(
                name="total_seconds_last_epoch",
                ipt=None,
                val=trainobj.total_seconds,
                data_specs=(NullSpace(), ''),
                dataset=trainobj.model.monitor._datasets[0])
    trainobj.first_callbacks_and_monitoring = True

def post_train(trainobj):
    """
    Code executed when the train was succesfully finalized (the
    training members requested the exit).
    """
    trainobj.model.monitor.training_succeeded = True

    if trainobj.save_freq > 0:
        trainobj.save()

def post_epoch(trainobj):
    """
    Code executed after each epoch. Returns True to continue or
    False to terminate.
    """
    trainobj.model.monitor.report_epoch()
    extension_continue = trainobj.run_callbacks_and_monitoring()
    freq = trainobj.save_freq
    if freq > 0 and trainobj.model.monitor.get_epochs_seen() % freq == 0:
        trainobj.save()
    if trainobj.algorithm is None:
        continue_learning = trainobj.model.continue_learning()
    else:
        continue_learning = trainobj.algorithm.continue_learning(trainobj.model)
    continue_learning = continue_learning and extension_continue
    assert continue_learning in [True, False, 0, 1]
    return continue_learning

def run_epoch(trainobj):
    """
    Runs an epoch. Returns True to continue or
    False to terminate.
    """

    if trainobj.first_callbacks_and_monitoring:
        trainobj.run_callbacks_and_monitoring()
        trainobj.first_callbacks_and_monitoring = False
        return True

    rval = True
    if trainobj.algorithm is None:
        rval = trainobj.model.train_all(dataset=trainobj.dataset)
        if rval is not None:
            raise ValueError("Model.train_all should not return " +
                             "anything. Use Model.continue_learning " +
                             "to control whether learning continues.")
        rval = post_epoch(trainobj)
    else:
        with log_timing(logger, None, level=logging.DEBUG,
                        callbacks=[trainobj.total_seconds.set_value]):
            with log_timing(logger, None, final_msg='Time this epoch:',
                            callbacks=[trainobj.training_seconds.set_value]):
                rval = trainobj.algorithm.train(dataset=trainobj.dataset)
            if rval is not None:
                raise ValueError("TrainingAlgorithm.train should not "
                                 "return anything. Use "
                                 "TrainingAlgorithm.continue_learning "
                                 "to control whether learning "
                                 "continues.")
            rval = post_epoch(trainobj)
    return rval

def main_loop(trainobj):
    """
    A main loop that behaves just like main loop in train.py.
    """
    pre_train(trainobj)
    while True:
        if not run_epoch(trainobj):
            break
    post_train(trainobj)
    trainobj.tear_down()

def main_debug_loop(trainobj):
    """
    A main loop that can be interrupted and restarted.
    """
    pre_train(trainobj)
    while True:
        if not run_epoch(trainobj):
            break
    post_train(trainobj)
    trainobj.tear_down()

def state_name(int_state):
    """
    Tell the status of the debugger in a string form.
    """
    if int_state == STS_LOADING:
        state_name = "Loading"
    elif int_state == STS_GROUND:
        state_name = "Ground"
    elif int_state == STS_STOPPED:
        state_name = "Stopped"
    elif int_state == STS_RUNNING:
        state_name = "Running"
    elif int_state == STS_PAUSED:
        state_name = "Paused"
    elif int_state == STS_REQ_STOP:
        state_name = "Request to stop"
    elif int_state == STS_REQ_PAUSE:
        state_name = "Request to pause"
    else:
        state_name = 'Unknown'
    return state_name
