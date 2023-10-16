CUDA Transfer Streams (CUts)
============================

This application is designed to launch intra-node transfer streams in an
adjustable way. It may trigger different types of CUDA transfers concurrently.
Each transfer is bound to a CUDA stream. Transfer buffers in main memory are
allocated (by default) on the proper NUMA node.


Dependencies
------------

* **cuda**
* **libnuma**


How to build CUts
-----------------

    % make


How to run CUts
---------------

    % ./cuts [ARGS...]

Arguments are :

    -d, --dtoh=<id>            Provide GPU id for Device to Host transfer.
    -h, --htod=<id>            Provide GPU id for Host to Device transfer.
    -i, --iter=<nb>            Specify the amount of iterations. [default: 100]
    -n, --no-numa-affinity     Do not make the transfer buffers NUMA aware.
    -p, --dtod=<id,id>         Provide comma-separated GPU ids to specify which
                               pair of GPUs to use for peer to peer transfer.
                               First id is the destination, second id is the
                               source.
    -s, --size=<bytes>         Specify the transfer size in bytes. [default:
                               1073741824]
    -?, --help                 Give this help list
        --usage                Give a short usage message
    -V, --version              Print program version


Disabling NVLink to test PCIe P2P between GPUs
----------------------------------------------

* Create a file */etc/modprobe.d/disable-nvlink.conf*
* Add the following line:

    options nvidia NVreg_NvLinkDisable=1


* reboot


Disabling ACS
-------------

PCIe Bandwidth may be lower in case of data transfers between two devices
connected to the same PCIe switch where Access Control Service (ACS) is enabled.
Ensuring ACS is disabled on all PCIe devices:

    for i in $(lspci | cut -f 1 -d " "); do setpci -v -s $i ecap_acs+6.w=0; done


Examples
--------

**PCIe P2P (NVLink disabled), both directions, between 2 GPUs connected to the same PCIe switch. ACS enabled:**

    % ./cuts --dtod=1,0 --dtod=0,1
    Launching P2P PCIe transfers from Device 0 to Device 1
    Launching P2P PCIe transfers from Device 1 to Device 0
    .........
    Completed.
    Transfer 0 - P2P transfers from device 0 to device 1: 12.037 GB/s  (8.92 seconds)
    Transfer 1 - P2P transfers from device 1 to device 0: 12.037 GB/s  (8.92 seconds)


**PCIe P2P (NVLink disabled), both directions, between 2 GPUs connected to the same PCIe switch. ACS disabled:**

    % ./cuts --dtod=1,0 --dtod=0,1
    Launching P2P PCIe transfers from Device 0 to Device 1
    Launching P2P PCIe transfers from Device 1 to Device 0
    ......
    Completed.
    Transfer 0 - P2P transfers from device 0 to device 1: 19.604 GB/s  (5.48 seconds)
    Transfer 1 - P2P transfers from device 1 to device 0: 19.448 GB/s  (5.52 seconds)


**Device to host direction with a single GPU:**

    % ./cuts --dtoh=0
    Launching Device to Host transfers with Device 0 (Host buffer allocated on NUMA node 3)
    .....
    Completed.
    Transfer 0 - Direct transfers with device 0 (Device to Host): 24.146 GB/s  (4.45 seconds)


**Host/Device transfers (both direction) from a single GPU:**

    % ./cuts --dtoh=0 --htod=0
    Launching Device to Host transfers with Device 0 (Host buffer allocated on NUMA node 3)
    Launching Host to Device transfers with Device 0 (Host buffer allocated on NUMA node 3)
    .......
    Completed.
    Transfer 0 - Direct transfers with device 0 (Device to Host): 15.650 GB/s  (6.86 seconds)
    Transfer 1 - Direct transfers with device 0 (Host to Device): 15.651 GB/s  (6.86 seconds)


**Device to host with two GPUs sharing same PCIe 4.0 switch (16x uplinks to root port):**

    %./cuts --dtoh=0 --dtoh=1
    Launching Device to Host transfers with Device 0 (Host buffer allocated on NUMA node 3)
    Launching Device to Host transfers with Device 1 (Host buffer allocated on NUMA node 3)
    .........
    Completed.
    Transfer 0 - Direct transfers with device 0 (Device to Host): 13.179 GB/s  (8.15 seconds)
    Transfer 1 - Direct transfers with device 1 (Device to Host): 13.179 GB/s  (8.15 seconds)


**Combining several transfer types with 8 GPUs:**

    % ./cuts --dtoh=0 --htod=1 --dtoh=2 --htod=3 --dtoh=4 --htod=5 --dtod=6,7
    Launching Device to Host transfers with Device 0 (Host buffer allocated on NUMA node 3)
    Launching Host to Device transfers with Device 1 (Host buffer allocated on NUMA node 3)
    Launching Device to Host transfers with Device 2 (Host buffer allocated on NUMA node 1)
    Launching Host to Device transfers with Device 3 (Host buffer allocated on NUMA node 1)
    Launching Device to Host transfers with Device 4 (Host buffer allocated on NUMA node 7)
    Launching Host to Device transfers with Device 5 (Host buffer allocated on NUMA node 7)
    Launching P2P PCIe transfers from Device 7 to Device 6
    ......
    Completed.
    Transfer 0 - Direct transfers with device 0 (Device to Host): 18.325 GB/s  (5.86 seconds)
    Transfer 1 - Direct transfers with device 1 (Host to Device): 18.319 GB/s  (5.86 seconds)
    Transfer 2 - Direct transfers with device 2 (Device to Host): 18.324 GB/s  (5.86 seconds)
    Transfer 3 - Direct transfers with device 3 (Host to Device): 18.320 GB/s  (5.86 seconds)
    Transfer 4 - Direct transfers with device 4 (Device to Host): 18.324 GB/s  (5.86 seconds)
    Transfer 5 - Direct transfers with device 5 (Host to Device): 18.320 GB/s  (5.86 seconds)
    Transfer 6 - P2P transfers from device 7 to device 6: 24.411 GB/s  (4.40 seconds)


HIP Version
-----------

To run on AMD GPUs, check the HIP version [HIts](https://github.com/jyvet/hits)
