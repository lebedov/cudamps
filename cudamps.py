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
    Manage MPS control daemons.

    Provides methods for querying, starting, and stopping MPS control daemons
    for multiple supported GPUs. Since no process states are stored in the cache
    instance, use of this class should not conflict with other tools that
    manipulate the control daemons (such as running the command-line management
    program).
    """

    def __init__(self):
        drv.init()

    def get_mps_ctrl_procs(self):
        """
        Find running MPS control daemons.

        Returns
        -------
        pids : list
            List of MPS control daemon process IDs.
        """

        try:
            out = subprocess.check_output(['pgrep', '-u',
                    str(os.getuid()), '-fx', '%s -d' % MPS_CTRL_PROG])
        except subprocess.CalledProcessError:
            return []
        else:
            return map(int, out.split())

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

    def get_mps_dir_by_dev(self, dev):
        """
        Find pipe directory for MPS control daemon process managing a device.

        Parameters
        ----------
        dev : int
            Device ID.

        Returns
        -------
        mps_dir : str
            Pipe directory associated with specified process. Returns None
            if the device is not found or is not associated with an MPS
            control daemon.
        """
        
        pids = self.get_mps_ctrl_procs()
        for pid in pids:
            devs = self.get_visible_devs(pid)
            if devs[0] == dev:
                return self.get_mps_dir(pid)
        return None

    def get_mps_proc_by_dev(self, dev):
        """
        Find MPS control daemon process managing a device.

        Parameters
        ----------
        dev : int
            Device ID.

        Returns
        -------
        pid : int
            Process ID. None is returned if no process is found.
        """

        pids = self.get_mps_ctrl_procs()
        for pid in pids:
            devs = self.get_visible_devs(pid)
            if devs[0] == dev:
                return pid
        return None

    def get_visible_devs(self, pid):
        """
        Find devices exposed to specified MPS server process.

        Parameters
        ----------
        pid : int
            MPS control daemon process ID.

        Returns
        -------
        devs : list
            List of device IDs.
        """

        data = self._get_proc_environ(pid)
        r = re.search('^CUDA_VISIBLE_DEVICES=(.*)$',
                      data, re.MULTILINE)
        try:
            devs = r.group(1)
        except:
            return None
        return map(int, devs.split(','))

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

    def start(self, dev, mps_dir=None):
        """
        Start MPS control daemon.

        Parameters
        ----------
        dev : int
            Device ID.
        mps_dir : str
            Pipe directory to be used by daemon. If no directory is
            specified, a new temporary directory is created. Logs are written to
            this directory too.
        """

        if mps_dir is None:
            mps_dir = tempfile.mkdtemp()

        # Map specified device number to the corresponding
        # number in the sequence of supported devices:
        try:
            i = self.get_supported_devs().index(dev)
        except:
            raise ValueError('device not supported')
        else:
            if self.get_mps_proc_by_dev(dev):
                raise ValueError('process associated with device %s exists' % dev)
            env = os.environ
            env['CUDA_VISIBLE_DEVICES'] = str(i)
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

    def start_all(self):
        """
        Start MPS control daemons for all supported devices.
        """

        devs = self.get_supported_devs()
        for i in devs:
            self.start(i)

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

    def stop_all(self, clean=False):
        """
        Stop all running MPS control daemons.

        Parameters
        ----------
        clean : bool
            If True, delete the pipe directories associated with the daemons.
        """
        
        pids = self.get_mps_ctrl_procs()
        for pid in pids:
            self.stop(pid, clean)
