

r"""
`torch.distributed.launch` is a module that spawns up multiple distributed
training processes on each of the training nodes.
The utility can be used for single-node distributed training, in which one or
more processes per node will be spawned. The utility can be used for either
CPU training or GPU training. If the utility is used for GPU training,
each distributed process will be operating on a single GPU. This can achieve
well-improved single-node training performance. It can also be used in
multi-node distributed training, by spawning up multiple processes on each node
for well-improved multi-node distributed training performance as well.
This will especially be benefitial for systems with multiple Infiniband
interfaces that have direct-GPU support, since all of them can be utilized for
aggregated communication bandwidth.
In both cases of single-node distributed training or multi-node distributed
training, this utility will launch the given number of processes per node
(``--nproc_per_node``). If used for GPU training, this number needs to be less
or euqal to the number of GPUs on the current system (``nproc_per_node``),
and each process will be operating on a single GPU from *GPU 0 to
GPU (nproc_per_node - 1)*.
**How to use this module:**
1. Single-Node multi-process distributed training
::
    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)
2. Multi-Node multi-process distributed training: (e.g. two nodes)
Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*
::
    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)
Node 2:
::
    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)
3. To look up what optional arguments this module offers:
::
    >>> python -m torch.distributed.launch --help
**Important Notices:**
1. This utilty and multi-process distributed (single-node or
multi-node) GPU training currently only achieves the best performance using
the NCCL distributed backend. Thus NCCL backend is the recommended backend to
use for GPU training.
2. In your training program, you must parse the command-line argument:
``--local_rank=LOCAL_PROCESS_RANK``, which will be provided by this module.
If your training program uses GPUs, you should ensure that your code only
runs on the GPU device of LOCAL_PROCESS_RANK. This can be done by:
Parsing the local_rank argument
::
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument("--local_rank", type=int)
    >>> args = parser.parse_args()
Set your device to local rank using either
::
    >>> torch.cuda.set_device(arg.local_rank)  # before your code runs
or
::
    >>> with torch.cuda.device(arg.local_rank):
    >>>    # your code to run
3. In your training program, you are supposed to call the following function
at the beginning to start the distributed backend. You need to make sure that
the init_method uses ``env://``, which is the only supported ``init_method``
by this module.
::
    torch.distributed.init_process_group(backend='YOUR BACKEND',
                                         init_method='env://')
4. In your training program, you can either use regular distributed functions
or use :func:`torch.nn.parallel.DistributedDataParallel` module. If your
training program uses GPUs for training and you would like to use
:func:`torch.nn.parallel.DistributedDataParallel` module,
here is how to configure it.
::
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[arg.local_rank],
                                                      output_device=arg.local_rank)
Please ensure that ``device_ids`` argument is set to be the only GPU device id
that your code will be operating on. This is generally the local rank of the
process. In other words, the ``device_ids`` needs to be ``[args.local_rank]``,
and ``output_device`` needs to be ``args.local_rank`` in order to use this
utility
5. Another way to pass ``local_rank`` to the subprocesses via environment variable
``LOCAL_RANK``. This behavior is enabled when you launch the script with
``--use_env=True``. You must adjust the subprocess example above to replace
``args.local_rank`` with ``os.environ['LOCAL_RANK']``; the launcher
will not pass ``--local_rank`` when you specify this flag.
.. warning::
    ``local_rank`` is NOT globally unique: it is only unique per process
    on a machine.  Thus, don't use it to decide if you should, e.g.,
    write to a networked filesystem.  See
    https://github.com/pytorch/pytorch/issues/12042 for an example of
    how things can go wrong if you don't do this correctly.
"""


import sys
import subprocess
import os
import socket
from argparse import ArgumentParser, REMAINDER

import torch


# def parse_args():
#     """
#     Helper function parsing the command line options
#     @retval ArgumentParser
#     """
#     parser = ArgumentParser(description="PyTorch distributed training launch "
#                                         "helper utilty that will spawn up "
#                                         "multiple distributed processes")
#
#     # Optional arguments for the launch helper
#     parser.add_argument("--nnodes", type=int, default=1,
#                         help="The number of nodes to use for distributed "
#                              "training")
#     parser.add_argument("--node_rank", type=int, default=0,
#                         help="The rank of the node for multi-node distributed "
#                              "training")
#     parser.add_argument("--nproc_per_node", type=int, default=1,
#                         help="The number of processes to launch on each node, "
#                              "for GPU training, this is recommended to be set "
#                              "to the number of GPUs in your system so that "
#                              "each process can be bound to a single GPU.")
#     parser.add_argument("--master_addr", default="127.0.0.1", type=str,
#                         help="Master node (rank 0)'s address, should be either "
#                              "the IP address or the hostname of node 0, for "
#                              "single node multi-proc training, the "
#                              "--master_addr can simply be 127.0.0.1")
#     parser.add_argument("--master_port", default=29500, type=int,
#                         help="Master node (rank 0)'s free port that needs to "
#                              "be used for communciation during distributed "
#                              "training")
#
#     # positional
#     parser.add_argument("training_script", type=str,
#                         help="The full path to the single GPU training "
#                              "program/script to be launched in parallel, "
#                              "followed by all the arguments for the "
#                              "training script")
#
#     # rest from the training program
#     parser.add_argument('training_script_args', nargs=REMAINDER)
#     return parser.parse_args()


def parse_args():
    """解析命令行参数

    返回：
        ArgumentParser对象，包含所有解析后的参数
    """
    parser = ArgumentParser(description="PyTorch分布式训练启动工具，用于在单节点或多节点环境中启动多个分布式进程")

    # 启动器专用参数
    parser.add_argument("--nnodes", type=int, default=1,
                        help="参与训练的节点总数（多节点训练时需要设置）")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="当前节点在多节点训练中的全局排名（从0开始）")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="每个节点启动的进程数（GPU训练时应设置为显卡数量）")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="主节点地址（单节点训练时使用127.0.0.1）")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="主节点通信端口（默认29500）")

    # 训练脚本参数
    parser.add_argument("training_script", type=str,
                        help="要执行的训练脚本完整路径")
    parser.add_argument('training_script_args', nargs=REMAINDER,
                        help="传递给训练脚本的参数列表")

    return parser.parse_args()


def main():
    """主函数，处理参数并启动子进程

     流程：
     1. 解析命令行参数
     2. 设置分布式环境变量
     3. 根据参数启动指定数量的子进程
     4. 等待所有子进程完成
     """
    args = parse_args()

    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        cmd = [args.training_script] + args.training_script_args

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=process.args)


if __name__ == "__main__":
    main()