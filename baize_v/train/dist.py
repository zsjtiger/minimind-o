import os
import random

import numpy as np
import torch
import torch.distributed as dist


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def init_distributed_mode():
    """torchrun 启动时初始化进程组：有 GPU 用 NCCL，否则用 gloo（便于 CPU 调试）。"""
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非分布式模式
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
