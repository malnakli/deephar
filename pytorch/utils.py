import torch

def tensor_to_numpy(tensor):
    if torch.cuda.is_available():
        return tensor.cpu().numpy()
    
    return tensor.numpy()
