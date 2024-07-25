import pytest
import torch
from src.warp_implementation import warp_update, slerp

def test_warp_update():
    theta_init = torch.tensor([1.0, 2.0, 3.0])
    theta_m = torch.tensor([1.5, 2.5, 3.5])
    theta_ema = torch.tensor([1.2, 2.2, 3.2])
    r_beta = torch.tensor([0.1, 0.2, 0.3])
    eta = 0.1
    mu = 0.9

    new_theta, new_theta_ema = warp_update(theta_init, theta_m, theta_ema, r_beta, eta, mu)

    assert torch.allclose(new_theta, torch.tensor([1.05, 2.05, 3.05]), atol=1e-6)
    assert torch.allclose(new_theta_ema, torch.tensor([1.185, 2.185, 3.185]), atol=1e-6)

def test_slerp():
    theta_init = torch.tensor([1.0, 0.0])
    theta_m_list = [torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.0])]
    lam = 0.5

    result = slerp(theta_init, theta_m_list, lam)

    # Ожидаемый результат: точка на большом круге, равноудаленная от всех трех точек
    expected = torch.tensor([0.0, 1.0]) / torch.sqrt(torch.tensor(2.0))
    assert torch.allclose(result, expected, atol=1e-6)

def test_slerp_identity():
    theta = torch.tensor([1.0, 2.0, 3.0])
    result = slerp(theta, [theta], 0.5)
    assert torch.allclose(result, theta, atol=1e-6)

def test_slerp_extreme_lambda():
    theta_init = torch.tensor([1.0, 0.0])
    theta_m = torch.tensor([0.0, 1.0])
    
    result_0 = slerp(theta_init, [theta_m], 0.0)
    result_1 = slerp(theta_init, [theta_m], 1.0)
    
    assert torch.allclose(result_0, theta_init, atol=1e-6)
    assert torch.allclose(result_1, theta_m, atol=1e-6)