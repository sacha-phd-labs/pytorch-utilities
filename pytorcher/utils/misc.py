import torch
import hashlib

def tensor_hash(t: torch.Tensor, format='sha256') -> str:
    t = t.detach().contiguous().cpu()   # remove grad, ensure contiguous memory
    if format == 'sha256':
        return hashlib.sha256(t.numpy().tobytes()).hexdigest()
    elif format == 'md5':
        return hashlib.md5(t.numpy().tobytes()).hexdigest()
    elif format == 'sha1':
        return hashlib.sha1(t.numpy().tobytes()).hexdigest()
    elif format == 'int':
        return int(hashlib.sha256(t.numpy().tobytes()).hexdigest(), 16)
    else:
        raise ValueError(f"Unsupported hash format: {format}")