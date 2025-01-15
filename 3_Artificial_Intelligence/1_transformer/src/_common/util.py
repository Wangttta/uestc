import os
import json
import torch.nn as nn


def get_abs_path(root=None, rel_path=None):
    abs_path = os.path.join(os.getcwd(), "3_Artificial_Intelligence", "transformer", "src") if root is None else root 
    if rel_path is not None:
        abs_path = os.path.join(abs_path, rel_path)
    return abs_path


def log_args(args, path):
    args_dict = vars(args)
    args_dict["device"] = str(args_dict["device"])
    output_file = os.path.join(path, "args.json")
    with open(output_file, 'w') as f:
        json.dump(args_dict, f, indent=4)


def he_initialization(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.kaiming_uniform_(m.weight.data)
