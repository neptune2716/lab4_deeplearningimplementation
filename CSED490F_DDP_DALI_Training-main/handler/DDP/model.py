import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def model_to_DDP(model):
    ''' Problem 4: model to DDP
    (./handler/DDP/model.py)
    Implement model_to_DDP function.
    model_to_DDP function is used to transfer model to DDP, SIMILAR with DP.
    Be careful for set devices. Set profer device id is important part in DDP.
    '''
    if not torch.cuda.is_available():
        raise RuntimeError("DDP model conversion requires CUDA availability")

    device_id = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_id}")
    model = model.to(device)

    # Each process holds a single GPU, so pin the module to that device id
    return DDP(model, device_ids=[device_id], output_device=device_id)