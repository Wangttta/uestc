import random
import numpy as np
import torch

from env.mumec import MUMEC
from algo.model import Model
from common.arguments import get_args


if __name__ == '__main__':

    # 1. Random parameters
    args = get_args()
    args.continuous_actions = False

    # 2. Init environment
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    env = MUMEC(args)

    # 3. Init resolver model and run
    model = Model(args=args, env=env)
    model.run()
