import carn_optional_v1
import torch
carn_optional_v1.compare_times([8, 16, 32, 64, 128, 256, 512, 1024], [torch.device("cpu"), torch.device("cuda")] )