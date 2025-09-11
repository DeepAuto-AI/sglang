import nvtx
import torch

with nvtx.annotate(message="Score candidate visuals", color="green"):
    a = torch.tensor([1, 2, 3], device="cuda")
    a = a + 1
    print(a)

