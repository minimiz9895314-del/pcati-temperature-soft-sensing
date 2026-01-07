import numpy as np
from dtc import DynamicTemperatureCompensation
from bgct import BayesianGatedConvTransformer
from utils import generate_synthetic_data, plot_results

# Generate synthetic data
Tg, T_meas = generate_synthetic_data(500)

# DTC
dtc = DynamicTemperatureCompensation()
T_comp = dtc.compensate(T_meas)

# Load BGCT model (assume trained)
model = BayesianGatedConvTransformer(input_dim=1)
model.load_state_dict(torch.load('bgct_model.pth'))
model.eval()

# Inference with uncertainty (Monte Carlo dropout)
seq_len = 50
predicted_means = []
predicted_stds = []
with torch.no_grad():
    for i in range(len(T_comp) - seq_len):
        x = torch.tensor(T_comp[i:i+seq_len, np.newaxis, np.newaxis]).float()  # [1, seq_len, 1]
        means = []
        vars_ = []
        for _ in range(50):  # T=50 samples
            mean, var = model(x, training=True)  # Dropout on
            means.append(mean.item())
            vars_.append(var.item())
        pred_mean = np.mean(means)
        pred_var = np.mean(vars_) + np.var(means)
        predicted_means.append(pred_mean)
        predicted_stds.append(np.sqrt(pred_var))

# Pad beginning with NaNs or zeros for plotting
predicted_means = np.concatenate([np.full(seq_len, np.nan), predicted_means])
predicted_stds = np.concatenate([np.full(seq_len, np.nan), predicted_stds])

plot_results(Tg, T_meas, T_comp, predicted_means, predicted_stds)