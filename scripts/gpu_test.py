import sys, torch
sys.path.insert(0, ".")
from probes.covariance_probe import CovarianceProbe

device = torch.device("cuda")
probe = CovarianceProbe(d_model=768).to(device)
print("Params device:", next(probe.parameters()).device)
print("Buffer device:", probe._eye.device)

x = torch.randn(32, 2, 2, 256, 768).to(device)
y = torch.randint(0, 2, (32,)).to(device)
out = probe(x)
print("Output:", out.shape)
loss = torch.nn.CrossEntropyLoss()(out, y)
loss.backward()
print("GPU training OK, loss=", loss.item())
