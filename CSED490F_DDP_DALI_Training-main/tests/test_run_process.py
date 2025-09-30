import os
import sys

import torch.multiprocessing as mp

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from handler.DDP.utils import run_process


class DummyArgs:
    def __init__(self, num_gpu, queue):
        self.num_gpu = num_gpu
        self.queue = queue


def _worker(rank, args):
    # push rank to queue so the parent process can verify all workers ran
    args.queue.put(rank)


def main():
    mp.set_start_method("spawn", force=True)
    queue = mp.Queue()
    args = DummyArgs(num_gpu=2, queue=queue)
    try:
        run_process(_worker, args)
        ranks = [queue.get(timeout=5) for _ in range(args.num_gpu)]
        assert sorted(ranks) == list(range(args.num_gpu))
    except AssertionError:
        print("test_run_process failed!", flush=True)
        raise
    print("test_run_process passed!", flush=True)


if __name__ == "__main__":
    main()
