#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DebuggerProxy class replaces the debugger on a GUI machine that
connects to a remote debugger.


@author: Nicu Tofan <nicu.tofan@gmail.com>
"""
import cPickle
import logging
logger = logging.getLogger(__name__)

import zmq
import time
from PyQt4 import QtCore

from learn_spot.gui import debugger

ERR_SIGNAL = QtCore.SIGNAL("error(QString)")

class Runnable(QtCore.QRunnable):
    """
    The worker thread for Debugger.
    """
    def __init__(self, debugger, socket):
        """
        Constructor
        """
        super(Runnable, self).__init__()
        self.debugger = debugger
        self.setAutoDelete(True)
        self.stop = False
        self.socket = socket

    def run(self):
        """
        Build-in method.
        """
        while not self.stop:
            message = self.socketrecv()
            try:
                # interpret the result
                response = cPickle.loads(message)

                if not isinstance(response, dict):
                    debugger.forward_error(
                              "Remote send unexpected message type: " + str(response.__class__))
                elif not response.has_key('type'):
                    debugger.forward_error(
                              "Remote send unexpected message: " + str(response))
                else:
                    self.debugger.forward_message(response)
            except Exception:
                msg = 'Failed to process broadcasted message'
                logger.debug(msg, exc_info=True)
                self.debugger.forward_error(msg)


class DebuggerProxy(QtCore.QObject):
    """
    The class represents a debugger running on a remote machine.

    Parameters
    ----------
    address : string
        The IP address on which a DebuggerPublisher process is listening.

    req_port : int
        The port number on which a DebuggerPublisher process is listening.
        The port is used to send requests and commands to the remote instance.

    pub_port : int
        The port number on which a DebuggerPublisher process is listening.
        The port is used to monitor the remote instance.

    Signals
    -------

    alive(bool)
        Tell if alive or not.
    debug_end(yaml_file)
        Returning to ground state.
    debug_start(yaml_file)
        A file was succesfully loaded into the debugger.
    debug_run()
        Entering paused state.
    debug_paused()
        Entering paused state.
    debug_stopped()
        Entering stopped state.
    error(message)
        An error happened.

    """
    def __init__(self, address='127.0.0.1', req_port=5955, pub_port=5956):
        """
        Constructor
        """
        super(DebuggerProxy, self).__init__()
        self.last_known_status = None

        self.address = 'tcp://%s' % address

        assert req_port != pub_port
        assert req_port > 1024 and req_port < 65536
        self.req_port = req_port
        assert pub_port > 1024 and pub_port < 65536
        self.pub_port = pub_port

        # prepare the network
        self.context = zmq.Context()
        self.reconnect()

        # hartbeat
        self.timer_hart_beat = self.startTimer(1)
        self.last_hartbeat_ok = False

    def reconnect(self):
        address_template = self.address + ':%d'
        self.req_sock = self.context.socket(zmq.REQ)
        self.req_sock.connect(address_template % self.req_port)
        self.req_sock.setsockopt(zmq.LINGER, 100)
        self.req_sock.setsockopt(zmq.RCVTIMEO, 100)
        self.req_sock.setsockopt(zmq.SNDTIMEO, 100)
        self.pub_sock = self.context.socket(zmq.SUB)
        self.pub_sock.setsockopt(zmq.SUBSCRIBE, "")
        self.pub_sock.connect(address_template % self.pub_port)

    def disconnect(self):
        address_template = self.address + ':%d'
        if self.req_sock:
            self.req_sock.unbind(address_template % self.req_port)
        if self.pub_sock:
            self.pub_sock.unbind(address_template % self.pub_port)

    def timerEvent(self, event):
        """
        """
        self.killTimer(self.timer_hart_beat )
        if self.context is None:
            self.context = zmq.Context()
            self.reconnect()

        if self.send_basic_command('status'):
            self.emit(QtCore.SIGNAL("alive"), True)
            next_after = 2000
            if not self.last_hartbeat_ok:
                self.req_sock.setsockopt(zmq.LINGER, 5000)
                self.req_sock.setsockopt(zmq.RCVTIMEO, 5000)
                self.req_sock.setsockopt(zmq.SNDTIMEO, 5000)
            self.last_hartbeat_ok = True
        else:
            print 'not alive'
            self.disconnect()
            self.reconnect()
            self.last_hartbeat_ok = False
            next_after = 100
        self.timer_hart_beat = self.startTimer(next_after)

    def is_running(self):
        """
        Tell if the debugger is in initialized state or not.

        Sends the `status` command to update the local status.
        """
        self.send_basic_command('status')
        return self.last_known_status == debugger.STS_RUNNING

    def unload_file(self):
        """
        Terminates debugging session.

        Sends the `unload_file` command.
        """
        self.send_basic_command('unload_file')

    def load_file(self, fname):
        """
        Starts debugging given file.

        Parameters
        ----------
        fname : str
            The path to the file to load on the remote machine.
            Top level object must be a Train object.
            The file is NOT uploaded from local machine.

        Returns
        -------
        init_ok : bool
            True if the file was succesfully loaded, False otherwise

        """
        self.send_basic_command('load_file', {'file': fname})
        return self.last_known_status != debugger.STS_GROUND

    def dbg_run(self, num_steps=None):
        """
        Slot that instructs the debugger to run a number of steps.

        Parameters
        ----------
        num_steps : int
            Number of epochs to perform. Pass None to run without interuption.

        """
        self.send_basic_command('run', {'steps': num_steps})

    def dbg_stop(self):
        """
        Slot that instructs the debugger to terminate after current epoch.

        """
        self.send_basic_command('stop')

    def dbg_pause(self):
        """
        Slot that instructs the debugger to pause after current epoch.

        """
        self.send_basic_command('pause')

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

    def forward_error(self, message):
        """
        """
        self.emit(ERR_SIGNAL, message)

    def forward_message(self, response):
        """
        """
        resp_type = response['type']

        if resp_type == 'debug_start':
            self.last_known_status = response['state']
            self.emit(QtCore.SIGNAL("debug_start"), response['message'])
        elif resp_type == 'debug_end':
            self.last_known_status = response['state']
            self.emit(QtCore.SIGNAL("debug_end"), response['message'])
        elif resp_type == 'debug_paused':
            self.last_known_status = response['state']
            self.emit(QtCore.SIGNAL("debug_paused"))
        elif resp_type == 'debug_run':
            self.last_known_status = response['state']
            self.emit(QtCore.SIGNAL("debug_run"))
        elif resp_type == 'debug_stopped':
            self.last_known_status = response['state']
            self.emit(QtCore.SIGNAL("debug_stopped"))
        elif resp_type == 'state':
            self.last_known_status = response['state']
            self.emit(QtCore.SIGNAL("debug_state_change(int,int)"), 
                      response['oldstate'], 
                      self.last_known_status)
        elif resp_type == 'error':
            self.emit(ERR_SIGNAL, response['message'])
        else:
            self.emit(ERR_SIGNAL, 'Unknown message type: ' + resp_type)

        self.updated_state(response['state'])

    def updated_state(self, new_state):
        """
        """
        if new_state != self.last_known_status:
            self.last_known_status = new_state
            if new_state == debugger.STS_STOPPED:
                self.emit(QtCore.SIGNAL("debug_stopped()"))
            elif new_state == debugger.STS_RUNNING:
                self.emit(QtCore.SIGNAL("debug_run()"))
            elif new_state == debugger.STS_PAUSED:
                self.emit(QtCore.SIGNAL("debug_paused()"))
            elif new_state == debugger.STS_GROUND:
                self.emit(QtCore.SIGNAL("debug_end()"), "")
        logger.debug('State is now %s', debugger.state_name(new_state))

    def process_reply(self, command, response):
        """
        Reply messages have `type: reply`, a `state` representing the
        state of the debugger after the command and an optional
        `warning` that contains errors produced while executing the command.
        """
        if not response.has_key('state'):
            self.emit(ERR_SIGNAL,
                      "Remote send `reply` message type without `state`: " +
                      str(response))

        if response.has_key('warning'):
            self.emit(ERR_SIGNAL,
                      "Remote error: " + response['warning'])
        self.updated_state(response['state'])

    def send_basic_command(self, command, args=None):
        """
        All responses should have a `type` key to indentify the kind of
        response.
        """
        b_finalized = False
        try:
            while self.context is None:
                time.sleep(0.5)

            # prepare the request
            command = {'request': command}
            if args:
                command.update(args)
            message = cPickle.dumps(command)

            # send it and get a reply back
            self.req_sock.send(message)
            response = self.req_sock.recv()

            # interpret the result
            response = cPickle.loads(response)
            if not isinstance(response, dict):
                self.emit(ERR_SIGNAL,
                          "Remote send unexpected message type: " + str(response.__class__))
            elif not response.has_key('type'):
                self.emit(ERR_SIGNAL,
                          "Remote send unexpected message: " + str(response))
            else:
                resp_type = response['type']
                if resp_type == 'reply':
                    self.process_reply(command, response)
            b_finalized = True
        except zmq.ZMQError:
            self.emit(QtCore.SIGNAL("alive"), False)
        except Exception:
            msg = 'Failed to send basic command'
            self.emit(ERR_SIGNAL, msg)
            logger.debug(msg, exc_info=True)
        return b_finalized
