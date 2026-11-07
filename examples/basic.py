import torch
from kotflash import attention

q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16, requires_grad=True)
v = torch.randn_like(k, requires_grad=True)

out = attention(q, k, v, causal=True, backend="auto")
loss = out.float().square().mean()
loss.backward()
print("output:", tuple(out.shape), "loss:", loss.item())
