.. -*- rst -*-

CUDAMPS
=======

Package Description
-------------------
CUDAMPS is a Python interface to the `CUDA Multi-Process Service 
<https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf>`_.  
It supports creation and management of MPS control daemons for multiple GPUs.

Prerequisites
-------------
* A Tesla or Quadro GPUs with compute capability 3.5 or later.
* `CUDA <http://www.nvidia.com/object/cuda_home_new.html>`_ 6.5 or later.
* `PyCUDA <http://mathema.tician.de/software/pycuda/>`_.
* `Pytools <https://pypi.python.org/pypi/pytools>`_.

Installation
------------
The package may be installed as follows: ::

    pip install cudamps

Development
-----------
The latest release of the package may be obtained from
`GitHub <https://github.com/lebedov/cudamps>`_.

Author
------
See the included AUTHORS.rst file for more information.

License
-------
This software is licensed under the
`BSD License <http://www.opensource.org/licenses/bsd-license>`_.
See the included LICENSE.rst file for more information.
