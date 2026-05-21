import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(num_samples=1000, ms=0.01, cs=500, As=0.001, hconv=50, heff=30, Te=300, dt=1.0):
    """
    Generate synthetic thermocouple data based on paper's heat balance equation.
    """
    tau = (ms * cs) / ((hconv + heff) * As)
    Tg = 300 + 800 * np.sin(np.linspace(0, 10 * np.pi, num_samples)) + np.random.normal(0, 10, num_samples)
    T = np.zeros(num_samples)
    T[0] = Te
    for t in range(1, num_samples):
        dT = dt / tau * ((Tg[t-1] - T[t-1]) * (hconv / (hconv + heff)) + (Te - T[t-1]) * (heff / (hconv + heff)))
        T[t] = T[t-1] + dT + np.random.normal(0, 1)
    return Tg, T  # True internal, measured wall

def plot_results(true, measured, compensated, predicted_mean, predicted_std):
    plt.figure(figsize=(12, 6))
    t = np.arange(len(true))
    plt.plot(t, true, label='True Internal Temp')
    plt.plot(t, measured, label='Measured Wall Temp')
    plt.plot(t, compensated, label='DTC Compensated')
    plt.plot(t, predicted_mean, label='BGCT Predicted Mean')
    plt.fill_between(t, predicted_mean - 2*predicted_std, predicted_mean + 2*predicted_std, alpha=0.3, label='95% CI')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature Soft Sensing Results')
    plt.savefig('results.png')
    plt.show()