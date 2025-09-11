import torch
import decord

decord.bridge.set_bridge("torch")
# vr = decord.VideoReader("/home/geon/86CxyhFV9MI.mp4", ctx=decord.gpu(0))
vr = decord.VideoReader("86CxyhFV9MI.mp4", ctx=decord.gpu(0))

print(vr.get_batch(torch.tensor([1000, 2000])))
