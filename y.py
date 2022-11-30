import torch
from torch.special import expm1

from einops import rearrange, repeat, reduce

@torch.jit.script
def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))


def get_sampling_timesteps(self, batch, *, device):
    times = torch.linspace(1., 0., 10, device = device)
    times = repeat(times, 't -> b t', b = batch)
    times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
    times = times.unbind(dim = -1)
    return times


timesteps = get_sampling_timesteps(4, 1, device='cpu')

for step in timesteps:
    print(step)
    print(beta_linear_log_snr(step))