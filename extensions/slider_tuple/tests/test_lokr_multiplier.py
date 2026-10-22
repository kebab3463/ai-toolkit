"""toolkit/models/lokr.py patch: LoKr applies one multiplier per sample, like LoRA (sliders need it)."""

import torch

from toolkit.models.lokr import LokrModule


class Net:
    """The network attributes a module's forward reads."""
    is_lorm = False
    is_active = True
    is_merged_in = False
    _multiplier = 1.0
    torch_multiplier = torch.tensor([1.0])


def lokr_layer(seed=0):
    torch.manual_seed(seed)
    net = Net()
    module = LokrModule("x", torch.nn.Linear(16, 32), multiplier=1.0, lora_dim=4, alpha=4, network=net, factor=4)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0, 0.5)  # nonzero update (the default init leaves it zero)
    module.apply_to()
    return module, net


def run(module, net, x, multipliers):
    net.torch_multiplier = torch.tensor(multipliers)
    return module(x)


def test_per_sample_multipliers_match_separate_runs():
    module, net = lokr_layer()
    x = torch.randn(2, 5, 16)
    both = run(module, net, x, [1.0, -1.0])
    first = run(module, net, x[:1], [1.0])
    second = run(module, net, x[1:], [-1.0])
    assert torch.allclose(both, torch.cat([first, second]), atol=1e-5)
    base = module.org_module[0](x)
    assert not torch.allclose(both - base, torch.zeros_like(base))  # +1 / -1 do not average to "off"


def test_repeated_batch_and_single_multiplier():
    module, net = lokr_layer()
    x = torch.randn(4, 3, 16)
    repeated = run(module, net, x, [0.5, -2.0])  # 2 multipliers over a 4-sample batch: each repeated twice
    expected = torch.cat([run(module, net, x[:2], [0.5]), run(module, net, x[2:], [-2.0])])
    assert torch.allclose(repeated, expected, atol=1e-5)
    single = run(module, net, x, [0.5])
    assert torch.allclose(single, run(module, net, x, [0.5, 0.5, 0.5, 0.5]), atol=1e-5)
