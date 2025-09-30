import os
import socket
import sys

import torch.distributed as dist
import torch.multiprocessing as mp

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from handler.DDP.utils import destroy_process, initialize_group, run_process


class InitArgs:
    def __init__(self, num_gpu, host, port):
        self.num_gpu = num_gpu
        self.host = host
        self.port = port


def _init_worker(proc_id, args):
    initialize_group(proc_id, args.host, args.port, args.num_gpu)

    assert dist.is_initialized()
    assert dist.get_world_size() == args.num_gpu
    assert dist.get_rank() == proc_id

    dist.barrier()
    destroy_process()


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    mp.set_start_method("spawn", force=True)
    if sys.platform.startswith("win"):
        print(
            "test_initialize_group skipped on Windows (gloo TCP backend unsupported)",
            flush=True,
        )
        return
    host = "localhost"
    port = _get_free_port()
    args = InitArgs(num_gpu=2, host=host, port=port)
    try:
        run_process(_init_worker, args)
    except AssertionError:
        print("test_initialize_group failed!", flush=True)
        raise
    print("test_initialize_group passed!", flush=True)


if __name__ == "__main__":
    main()
