#!/usr/bin/env python

"""
Demo of how to use cudamps.

Usage
-----
Run this demo as follows: ::

    mpiexec -np 1 python mpi4py_demo.py

Requirements
------------
* mpi4py (built against an MPI implementation that supports
  dynamic process management, e.g., OpenMPI)
* numpy
* pycuda
"""

import os
import shutil
import sys

from mpi4py import MPI
import numpy as np

import cudamps

def worker():
    comm = MPI.Comm.Get_parent()
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    from pycuda.elementwise import ElementwiseKernel
    
    kernel = ElementwiseKernel('double *y, double *x, double a',
                               'y[i] = a*x[i]')
    x_gpu = gpuarray.to_gpu(np.random.rand(2))
    y_gpu = gpuarray.empty_like(x_gpu)
    kernel(y_gpu, x_gpu, np.double(2.0))

    print 'I am process %d of %d on %s [x_gpu=%s, y_gpu=%s]' % \
        (rank, size, name, str(x_gpu.get()), str(y_gpu.get()))
    comm.Disconnect()

if __name__ == '__main__':
    script_file_name = os.path.basename(__file__)

    # The parent of the spawning process has a null communicator:
    if MPI.Comm.Get_parent() != MPI.COMM_NULL:
        worker()
    else:

        mps_man = cudamps.MultiProcessServiceManager()
        mps_man.start(0)
        mps_dir = mps_man.get_mps_dir_by_dev(0)
        print 'started MPS control daemon from launcher with directory %s' % mps_dir

        # Create lists of parameters:
        maxprocs = 3
        cmd_list = [sys.executable]*maxprocs
        args_list = [script_file_name]*maxprocs
        maxprocs_list = [1]*maxprocs
        info_list = [MPI.Info.Create() for i in xrange(maxprocs)]
        for i in xrange(maxprocs):
            info_list[i].Set('env', 'CUDA_MPS_PIPE_DIRECTORY=%s' % mps_dir)

        # Launch:
        comm = MPI.COMM_SELF.Spawn_multiple(cmd_list,
                                            args=args_list,
                                            maxprocs=maxprocs_list,
                                            info=info_list)
        comm.Disconnect()
        mps_man.stop(mps_man.get_mps_proc_by_dev(0))
        print 'stopped MPS control daemon from launcher'

        # Show the server log:
        print '--- server log ---'+'-'*(100-18)
        with open(os.path.join(mps_dir, 'server.log'), 'r') as f:
            print f.read().strip()
        print '-'*100

        # Clean up:
        shutil.rmtree(mps_dir)
