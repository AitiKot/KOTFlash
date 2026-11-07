import torch

from kotflash.attention import _blockwise_backward


def test_blockwise_backward_matches_sdpa_reference():
    torch.manual_seed(7)
    for causal in (False, True):
        q = torch.randn(1, 4, 7, 8, requires_grad=True)
        k = torch.randn(1, 2, 9, 8, requires_grad=True)
        v = torch.randn(1, 2, 9, 8, requires_grad=True)
        out_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), is_causal=causal
        )
        grad_out = torch.randn_like(out_ref)
        expected = torch.autograd.grad(out_ref, (q, k, v), grad_out)

        qf, kf, vf = q.detach(), k.detach(), v.detach()
        outs, lses = [], []
        for h in range(4):
            kh = h // 2
            scores = qf[0, h] @ kf[0, kh].T / (8 ** 0.5)
            if causal:
                mask = torch.arange(9)[None, :] <= torch.arange(7)[:, None]
                scores = scores.masked_fill(~mask, float("-inf"))
            lse = torch.logsumexp(scores.float(), dim=-1)
            outs.append(torch.softmax(scores.float(), dim=-1) @ vf[0, kh].float())
            lses.append(lse)
        out = torch.stack(outs).unsqueeze(0)
        lse = torch.stack(lses).unsqueeze(0)
        actual = _blockwise_backward(
            qf, kf, vf, out, lse, grad_out,
            causal=causal, scale=8 ** -0.5, q_block=3, kv_block=4,
        )
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)
