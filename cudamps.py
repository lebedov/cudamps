#!/usr/bin/env python

"""
Python interface to CUDA Multi-Process Service.
"""

# Copyright (c) 2013-2014, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

import os
import re
import subprocess
import sys
import tempfile

import pycuda.driver as drv
import pytools

MPS_CTRL_PROG = 'nvidia-cuda-mps-control'

class MultiProcessServiceManager(object):
    """
    Manage MPS server daemons.
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
            Pipe directory associated with specified process.
        """

        try:
            f = open('/proc/%i/environ' % pid, 'r') 
        except:
            return ''
        else:
            data = f.read().replace('\0', '\n')
            r = re.search('^CUDA_MPS_PIPE_DIRECTORY=(.*)$',
                          data, re.MULTILINE)
            try:
                return r.group(1)
            except:
                return ''

    def get_visible_devices(self, pid):
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

        try:
            f = open('/proc/%i/environ' % pid, 'r') 
        except:
            return ''
        else:
            data = f.read().replace('\0', '\n')
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
            specified, a new temporary directory is created.

        Notes
        -----
        Permits one to start more than one daemon for the same device.
        This probably shouldn't be allowed.
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
            env = os.environ
            env['CUDA_VISIBLE_DEVICES'] = str(i)
            env['CUDA_MPS_PIPE_DIRECTORY'] = mps_dir
            env['CUDA_MPS_LOG_DIRECTORY'] = mps_dir
            p = subprocess.Popen([MPS_CTRL_PROG, '-d'],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 env=env)
            out = p.communicate()[0]
            if 'An instance of this daemon is already running' in out:
                raise RuntimeError('running daemon already using %s' % mps_dir)

    def stop(self, pid):
        """
        Stop MPS control daemon.

        Parameters
        ----------
        pid : int
            MPS control daemon process ID.

        Notes
        -----
        The pipe directory associated with the stopped daemon is not
        deleted by this method.
        """

        mps_dir = self.get_mps_dir(pid)
        if mps_dir:
            env = os.environ
            env['CUDA_MPS_PIPE_DIRECTORY'] = mps_dir
            p = subprocess.Popen([MPS_CTRL_PROG], stdin=subprocess.PIPE)
            p.communicate('quit\n')
        else:
            raise ValueError('error stopping process %i' % pid)

