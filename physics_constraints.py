import torch

def energy_conservation_loss(pred, prev_pred, Tg, Te, ms, cs, As, hconv, heff, dt):
    dT_dt = (pred - prev_pred) / dt
    net_heat = ms * cs * dT_dt
    conv_heat = hconv * As * (Tg - pred)
    eff_heat = heff * As * (Te - pred)
    return torch.mean((net_heat - conv_heat - eff_heat) ** 2)

def steady_state_loss(pred, Tg, Te, hconv, heff):
    weighted_avg = (hconv * As * Tg + heff * As * Te) / (hconv * As + heff * As)
    return torch.mean((pred - weighted_avg) ** 2)

def temp_rate_loss(pred, prev_pred, Tg, ms, cs, As, hconv, heff, dt):
    dT_dt = (pred - prev_pred) / dt
    max_rate = (hconv * As + heff * As) / (ms * cs) * torch.abs(Tg - pred)
    violation = torch.relu(torch.abs(dT_dt) - max_rate)
    return torch.mean(violation ** 2)

def physics_loss(pred, prev_pred, Tg, Te, params, dt):
    ms, cs, As, hconv, heff = params
    l_energy = energy_conservation_loss(pred, prev_pred, Tg, Te, ms, cs, As, hconv, heff, dt)
    l_steady = steady_state_loss(pred, Tg, Te, hconv, heff)
    l_temp = temp_rate_loss(pred, prev_pred, Tg, ms, cs, As, hconv, heff, dt)
    return 0.33 * l_energy + 0.33 * l_steady + 0.34 * l_temp