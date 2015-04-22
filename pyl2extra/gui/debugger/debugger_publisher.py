#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implements the publisher used in remote deployments.

@author: Nicu Tofan <nicu.tofan@gmail.com>
"""
import cPickle
import logging
logger = logging.getLogger(__name__)

import zmq
from learn_spot.gui.debugger import Debugger
from learn_spot.gui.signal_handler import SignalHandler

class DebuggerPublisher(SignalHandler):
    """
    This is a signal handler used on a remote, headless machine.
    
    The messages the debugger produces are published using 
    publisher-subscriber pattern. An additional 
    functionality is that the class allows commands to be send to the
    debugger using send-request pattern, with this class being the server.

    Parameters
    ----------
    address : string
        The IP addresses of the interfaces on which the instance should listen.

    req_port : int
        The port number to be used to service request.

    pub_port : int
        The port number to be used to publish updates.
    """
    def __init__(self, debugger=None, address='*', req_port=5955, pub_port=5956):
        """
        Constructor.
        """
        super(DebuggerPublisher, self).__init__()

        self.address = 'tcp://%s' % address
        address_template = self.address + ':%d'

        assert req_port != pub_port
        assert req_port > 1024 and req_port < 65536
        self.req_port = req_port
        assert pub_port > 1024 and pub_port < 65536
        self.pub_port = pub_port

        # prepare the network
        self.context = zmq.Context()
        self.req_sock = self.context.socket(zmq.REP)
        self.req_sock.bind(address_template % self.req_port)
        self.pub_sock = self.context.socket(zmq.PUB)
        self.pub_sock.bind(address_template % self.pub_port)

        # create a local debugger or use the one that was provided
        if debugger is None:
            self.debugger = Debugger(self)
        else:
            self.debugger = debugger

        # used to capture errors that happen durring requests
        self.error_catcher = None
        self.in_request = False

    def debug_start(self, yaml_file):
        """
        Slot that connects to debugger; raised when the debugger
        loaded a file.
        """
        logger.info('Debugger loaded: %s', yaml_file)
        self.publish('debug_start', yaml_file)

    def debug_end(self, yaml_file):
        """
        Slot that connects to debugger; raised when the debugger
        unloaded a file.
        """
        logger.info('Debugger unloaded %s', yaml_file)
        self.publish('debug_end', yaml_file)

    def debug_run(self):
        """
        Slot that connects to debugger; raised when the debugger
        starts a run.
        """
        logger.info('Debugger running')
        self.publish('debug_run')

    def debug_paused(self):
        """
        Slot that connects to debugger; raised when the debugger
        enters pause.
        """
        logger.info('Debugger paused')
        self.publish('debug_paused')

    def debug_stopped(self):
        """
        Slot that connects to debugger; raised when the debugger
        was stopped.
        """
        logger.info('Debugger stopped')
        self.publish('debug_stopped')

    def debug_error(self, message):
        """
        Slot that connects to debugger; raised when the debugger
        encounters an error.
        """
        logger.error('Debugger error: %s', message)
        if self.in_request:
            self.error_catcher = message
        else:
            self.publish('error', message)

    def debug_state_change(self, old_state, new_state):
        """
        Slot that connects to debugger; raised when the debugger
        changes the state.
        """
        message = cPickle.dumps({'type': 'state',
                                 'message': None, 
                                 'state': new_state,
                                 'oldstate': old_state})
        self.pub_sock.send(message)

    def publish(self, event_name, message=None):
        """
        Publish an event and message.
        """
        message = cPickle.dumps({'type': event_name,
                                 'message': message, 
                                 'state': self.debugger.state})
        self.pub_sock.send(message)

    def req_load_file(self, message):
        if message.has_key('file'):
            local_file = message['file']
            if message.has_key('content'):
                backup_file(local_file)
                with open(local_file, 'rb') as f:
                    f.write(message['content'])
            self.debugger.load_file(local_file)
            response = {'type': 'reply', 'state': self.debugger.state}
        else:
            msg_text = 'Request to load file without <file>'
            response = {'type': 'error', 'message': msg_text}
            logger.error(msg_text)
        return response

    def req_run(self, message):
        """
        Process a request to run a number of epochs.
        """
        try:
            steps = message['steps']
        except KeyError:
            steps = 1
        self.debugger.dbg_run(steps)
        response = {'type': 'reply', 'state': self.debugger.state}
        return response

    def process_request(self, message):
        """
        The messages processed by this methods are dictionaries.
        """
        req_type = message['request']
        logger.debug("Processing request: %s", req_type)

        # -- information -- #
        if req_type == 'status':
            response = {'type': 'reply', 'state': self.debugger.state}

        # -- commands -- #
        elif req_type == 'terminate':
            response = None
        elif req_type == 'unload_file':
            self.debugger.unload_file()
            response = {'type': 'reply', 'state': self.debugger.state}
        elif req_type == 'load_file':
            response = self.req_load_file(message)
        elif req_type == 'run':
            response = self.req_run(message)
        elif req_type == 'stop':
            self.debugger.dbg_stop()
            response = {'type': 'reply', 'state': self.debugger.state}
        elif req_type == 'pause':
            self.debugger.dbg_pause()
            response = {'type': 'reply', 'state': self.debugger.state}
        elif req_type == 'continue':
            self.debugger.dbg_continue()
            response = {'type': 'reply', 'state': self.debugger.state}

        # -- unknown -- #
        else:
            msg_text = 'Received unknown request <%s> in request slot' % req_type
            response = {'type': 'error', 'message': msg_text}
            logger.error(msg_text)

        logger.debug("Reply: %s", str(response))
        return response

    def run(self):
        """
        This is the command and control interface.
        """
        b_continue = True
        while b_continue:
            #  Wait for next request from client
            message = self.req_sock.recv()
            message = cPickle.loads(message)
            logger.debug("Received request: %s", str(message))

            if not isinstance(message, dict):
                msg_text = 'Received non-dict as message in request slot'
                response = {'type': 'error', 'message': msg_text}
                logger.error(msg_text)
            elif not message.has_key('request'):
                msg_text = 'Received message in request slot without <request>'
                response = {'type': 'error', 'message': msg_text}
                logger.error(msg_text)
            else:
                self.in_request = True
                self.error_catcher = None
                response = self.process_request(message)
                if not response:
                    # terminate the loop
                    response = {'type': 'reply', 'state': self.debugger.state}
                    b_continue = False
                if self.error_catcher:
                    response['warning'] = self.error_catcher
                    self.error_catcher = None
                self.in_request = False

            response = cPickle.dumps(response)
            self.req_sock.send(response)

import shutil
def backup_file(local_file):
    """
    back-up a file.
    
    TODO: this should allow a number of back-ups.
    """
    shutil.copy (local_file, local_file+'.bkp')
    