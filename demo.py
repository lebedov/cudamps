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

import atexit
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

    import pycuda.driver as drv
    drv.init()

    # Find maximum number of available GPUs:
    max_gpus = drv.Device.count()

    # Use modular arithmetic to avoid assigning a nonexistent GPU:
    n = rank % max_gpus
    dev = drv.Device(n)
    ctx = dev.make_context()
    atexit.register(ctx.pop)

    # Execute a kernel:
    import pycuda.gpuarray as gpuarray
    from pycuda.elementwise import ElementwiseKernel
    
    kernel = ElementwiseKernel('double *y, double *x, double a',
                               'y[i] = a*x[i]')
    x_gpu = gpuarray.to_gpu(np.random.rand(2))
    y_gpu = gpuarray.empty_like(x_gpu)
    kernel(y_gpu, x_gpu, np.double(2.0))

    print 'I am process %d of %d on CPU %s using GPU %s of %s [x_gpu=%s, y_gpu=%s]' % \
        (rank, size, name, n, max_gpus, str(x_gpu.get()), str(y_gpu.get()))
    comm.Disconnect()

if __name__ == '__main__':
    script_file_name = os.path.basename(__file__)

    # The parent of the spawning process has a null communicator:
    if MPI.Comm.Get_parent() != MPI.COMM_NULL:
        worker()
    else:

        mps_man = cudamps.MultiProcessServiceManager()
        mps_man.start()
        mps_dir = mps_man.get_mps_dir(mps_man.get_mps_ctrl_proc())
        print 'started MPS control daemon from launcher with directory %s' % mps_dir

        # Create lists of parameters:
        maxprocs = 3
        info = MPI.Info.Create()
        info.Set('env', 'CUDA_MPS_PIPE_DIRECTORY=%s' % mps_dir)

        # Launch:
        comm = MPI.COMM_SELF.Spawn(sys.executable,
                                   args=[script_file_name],
                                   maxprocs=maxprocs,
                                   info=info)
        comm.Disconnect()
        mps_man.stop(mps_man.get_mps_ctrl_proc())
        print 'stopped MPS control daemon from launcher'

        # Show the server log:
        print '--- server log ---'+'-'*(100-18)
        with open(os.path.join(mps_dir, 'server.log'), 'r') as f:
            print f.read().strip()
        print '-'*100

        # Clean up:
        shutil.rmtree(mps_dir)
