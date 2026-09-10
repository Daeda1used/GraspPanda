"""Exact chunk evaluation of the released PointRWKV matrix-state recurrence."""
import torch


def parallel_recurrence(r, k, v, w, u, chunk_size=32):
    """Inputs [B,H,T,D]; row-wise decay, native receptance and native bonus.

    Direct products avoid division by a cumulative decay, including zero decay.
    Work inside each fixed-size chunk is parallel; its outgoing matrix state is
    passed to the next chunk. This implements released code, not paper Eq. 4.
    """
    batch, heads, length, width = r.shape
    if not length or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('Nonempty sequence and positive integer chunk size required')
    if any(x.shape != r.shape for x in (k, v, w)) or u.shape != (heads, width):
        raise ValueError('Incompatible PointRWKV recurrence shapes')
    state = r.new_zeros(batch, heads, width, width)
    outputs = []
    for begin in range(0, length, chunk_size):
        end = min(begin + chunk_size, length)
        rc, kc, vc, wc = (x[:, :, begin:end] for x in (r, k, v, w))
        count = end - begin
        position = torch.arange(count, device=r.device)
        later = position[:, None] > position[None, :]
        # products[t,i,d] = product of w[j,d] over i < j <= t.
        factors = torch.where(later[None, None, :, :, None], wc.unsqueeze(-2), 1.)
        products = factors.cumprod(dim=2)
        previous = torch.cat((torch.ones_like(products[:, :, :1]), products[:, :, :-1]), dim=2)
        previous = previous * later[None, None, :, :, None]
        prefix = wc.cumprod(dim=2)
        before = torch.cat((torch.ones_like(prefix[:, :, :1]), prefix[:, :, :-1]), dim=2)
        incoming = torch.einsum('bhde,bhte->bhtd', state, kc) * before
        similarities = torch.einsum('bhie,bhte->bhti', vc, kc)
        internal = torch.einsum('bhti,bhtid,bhid->bhtd', similarities, previous, kc)
        bonus = kc * vc.sum(dim=-1, keepdim=True) * u[None, :, None]
        outputs.append(torch.sigmoid(rc) * (incoming + internal + bonus))
        state = prefix[:, :, -1, :, None] * state + torch.einsum('bhid,bhie->bhde', kc * products[:, :, -1], vc)
    return torch.cat(outputs, dim=2)
