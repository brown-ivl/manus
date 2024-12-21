import torch

def attach(array, isnumpy=True, device=torch.device("cpu")):
    if isnumpy:
        return torch.from_numpy(array).float().to(device)
    else:
        return array.to(device)

def detach(tensor, tonumpy=True):
    if tonumpy:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().cpu()