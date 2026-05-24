import torch


def to(obj, device: torch.device | str | None):
    """
    Move nested tensors to a device while leaving non-tensor objects unchanged.
    """
    if device is None:
        return obj
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(to(v, device) for v in obj)
    if isinstance(obj, list):
        return [to(v, device) for v in obj]
    return obj


def get_rc_tensor(rc_np, aatype):
    return torch.as_tensor(rc_np, device=aatype.device)[aatype]


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s, device=inds.device)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]
