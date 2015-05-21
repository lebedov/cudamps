#!/usr/bin/env python

"""
Python interface to CUDA Multi-Process Service.
"""

# Copyright (c) 2015, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

import os
import re
import sys
import tempfile

# Needed to support timeouts with Python 2.7:
if sys.version_info < (3, 0):
    import subprocess32 as subprocess
else:
    import subprocess

import pycuda.driver as drv
import pytools

MPS_CTRL_PROG = 'nvidia-cuda-mps-control'

class MultiProcessServiceManager(object):
    """
    Manage MPS control daemon.

    Provides methods for querying, starting, and stopping the MPS control daemon
    for multiple supported GPUs. Since no process states are stored in the cache
    instance, use of this class should not conflict with other tools that
    manipulate the control daemons (such as running the command-line management
    program).
    """

    def __init__(self):
        drv.init()

    def get_mps_ctrl_proc(self):
        """
        Find running MPS control daemon.

        Returns
        -------
        pid : int
            MPS control daemon process ID. If more than one daemon is found,
            the first is returned.
        """

        try:
            out = subprocess.check_output(['pgrep', '-u',
                    str(os.getuid()), '-fx', '%s -d' % MPS_CTRL_PROG])
        except subprocess.CalledProcessError:
            return None
        else:
            return int(out.split()[0])

    def _get_proc_environ(self, pid):
        """
        Retrieve environment of running control daemon.

        Parameters
        ----------
        pid : int
            MPS control daemon process ID.

        Returns
        -------
        data : str
            Process environment.
        """

        try:
            f = open('/proc/%i/environ' % pid, 'r')
        except:
            return ''
        else:
            return f.read().replace('\0', '\n')
        
    def get_mps_dir(self, pid):
        """
        Find pipe directory for MPS control daemon process.

        Parameters
        ----------
        pid : int
            MPS control daemon process ID.

        Returns
        -------
        mps_dir : str
            Pipe directory associated with specified process. Returns None
            if the process is not found or is not an MPS control daemon.
        """

        data = self._get_proc_environ(pid)
        r = re.search('^CUDA_MPS_PIPE_DIRECTORY=(.*)$',
                      data, re.MULTILINE)
        try:
            return r.group(1)
        except:
            return None

    @pytools.memoize_method
    def get_supported_devs(self):
        """
        Find local GPUs that support MPS.

        Returns
        -------
        devs : list
            List of IDs for devices that are either Tesla or Quadro and
            have a compute capability of at least 3.5.
        """

        result = []
        for i in xrange(drv.Device.count()):
            d = drv.Device(i)
            if d.compute_capability() >= (3, 5) and \
               re.search('Tesla|Quadro', d.name()):
                result.append(i)
        return result

    def start(self, mps_dir=None):
        """
        Start MPS control daemon.

        Parameters
        ----------
        mps_dir : str
            Pipe directory to be used by daemon. If no directory is
            specified, a new temporary directory is created. Logs are written to
            this directory too.
        """

        if mps_dir is None:
            mps_dir = tempfile.mkdtemp()

        env = os.environ
        env['CUDA_MPS_PIPE_DIRECTORY'] = mps_dir
        env['CUDA_MPS_LOG_DIRECTORY'] = mps_dir
        p = subprocess.Popen([MPS_CTRL_PROG, '-d'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             env=env)
        try:
            out = p.communicate(timeout=0.5)[0]
        except subprocess.TimeoutExpired:
            pass
        else:
            if 'An instance of this daemon is already running' in out:
                raise RuntimeError('running daemon already using %s' % mps_dir)

    def stop(self, pid, clean=False):
        """
        Stop MPS control daemon.

        Parameters
        ----------
        pid : int
            MPS control daemon process ID.
        clean : bool
            If True, delete the pipe directory associated with the daemon.
        """

        mps_dir = self.get_mps_dir(pid)
        if mps_dir:
            env = os.environ
            env['CUDA_MPS_PIPE_DIRECTORY'] = mps_dir
            p = subprocess.Popen([MPS_CTRL_PROG], stdin=subprocess.PIPE)
            p.communicate('quit\n')
        else:
            raise ValueError('error stopping process %i' % pid)
