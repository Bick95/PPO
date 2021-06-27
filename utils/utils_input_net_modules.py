import torch


def extract_params_from_structure(structure: list, index: int, key: str, vertical_dim: bool, default: int):
    # Takes the specification of the structure of a NN and returns a requested element from it
    if key in structure[index].keys():
        # Requested specification is provided
        if isinstance(structure[index][key], int):
            # Same specification for both vertical and horizontal case
            return structure[index][key]
        elif isinstance(structure[index][key], tuple):
            # Different specifications for vertical and horizontal axes respectively
            return structure[index][key][0 if vertical_dim else 1]
        else:
            # Assumed different specifications for vertical and horizontal axes respectively, just not in tuple format
            return tuple(structure[index][key])[0 if vertical_dim else 1]
    else:
        # Requested specification is not provided
        return default


def size_preserving_padding(i, k, d: int = 1, s: int = 1):
    # Can be generalized to: ((out_dim - 1)*stride - input_dim + kernel_size + (kernel_size-1)*(dilation-1)) / 2)
    return int(torch.floor(torch.tensor(((i - 1)*s - i + k + (k-1)*(d-1)) / 2)).numpy())


def out_dim(i, k, p, d: int = 1, s: int = 1):
    # Ref: https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338/2?u=bick95
    return int(torch.floor(torch.tensor((i + 2*p - k - (k - 1)*(d - 1)) / s + 1)).numpy())