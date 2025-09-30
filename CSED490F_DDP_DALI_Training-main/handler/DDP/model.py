import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def model_to_DDP(model):
    ''' Problem 4: model to DDP
    (./handler/DDP/model.py)
    Implement model_to_DDP function.
    model_to_DDP function is used to transfer model to DDP, SIMILAR with DP.
    Be careful for set devices. Set profer device id is important part in DDP.
    '''
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        device = torch.device(f"cuda:{device_id}")
        model = model.to(device)
        # Wrap model so gradients are synchronized across ranks for this device
        return DDP(model, device_ids=[device_id], output_device=device_id, broadcast_buffers=True)

    # CPU fallback primarily for testing environments without CUDA
    return DDP(model)