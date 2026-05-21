import numpy as np
from scipy.linalg import inv

class DynamicTemperatureCompensation:
    def __init__(self, ms=0.01, cs=500, As=0.001, hconv=50, hrad=20, k=20, Ac=1e-6, L=0.1, Te=300, dt=1.0):
        """
        Initialize DTC model parameters based on paper.
        - ms: mass of thermocouple head (kg)
        - cs: specific heat capacity (J/kg*K)
        - As: surface area (m^2)
        - hconv: convective heat transfer coeff (W/m^2*K)
        - hrad: radiative heat transfer coeff (W/m^2*K)
        - k: thermal conductivity (W/m*K)
        - Ac: cross-sectional area (m^2)
        - L: length (m)
        - Te: ambient temperature (K)
        - dt: time step (s)
        """
        self.ms = ms
        self.cs = cs
        self.As = As
        self.hconv = hconv
        self.hrad = hrad
        self.k = k
        self.Ac = Ac
        self.L = L
        self.Te = Te
        self.dt = dt
        
        # Calculate effective h_eff = h_rad + (k * Ac) / (L * As)
        self.heff = self.hrad + (self.k * self.Ac) / (self.L * self.As)
        
        # Initial time constant tau and gains k1, k2
        self.tau = (self.ms * self.cs) / (self.hconv * self.As + self.heff * self.As)
        self.k1 = (self.hconv * self.As) / (self.hconv * self.As + self.heff * self.As)
        self.k2 = self.heff * self.As / (self.hconv * self.As + self.heff * self.As)
        
        # Kalman filter matrices
        self.A = np.array([[1 - self.dt / self.tau, self.k1 * self.dt / self.tau],
                           [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q = np.eye(2) * 0.01  # Process noise
        self.R = 0.1  # Measurement noise
        self.P = np.eye(2)  # Initial covariance
        
        # State [T, Tg]
        self.x_hat = np.array([self.Te, self.Te])
        
        # RLS parameters
        self.theta = np.array([self.tau, self.k1, self.k2])  # [tau, k1, k2]
        self.P_rls = np.eye(3) * 1000  # Large initial covariance for RLS
        self.lambda_rls = 0.98  # Forgetting factor

    def kalman_step(self, y):
        """
        Kalman filter step to estimate states [T, Tg]
        - y: measured temperature T
        """
        # Prediction
        x_pred = self.A @ self.x_hat + np.array([self.k2 * self.dt / self.tau * self.Te, 0])
        P_pred = self.A @ self.P @ self.A.T + self.Q
        
        # Update
        K = P_pred @ self.H.T @ inv(self.H @ P_pred @ self.H.T + self.R)
        self.x_hat = x_pred + K * (y - self.H @ x_pred)
        self.P = (np.eye(2) - K @ self.H) @ P_pred
        
        return self.x_hat[1]  # Estimated Tg

    def rls_step(self, x1_k, x2_k, Te_k, x1_k1):
        """
        RLS step to update theta = [tau, k1, k2]
        - x1_k: current T
        - x2_k: current Tg
        - Te_k: current Te
        - x1_k1: previous T
        """
        phi = np.array([-x1_k1 / self.dt, x2_k / self.dt, Te_k / self.dt])
        y = x1_k
        
        # RLS update
        K_rls = self.P_rls @ phi / (self.lambda_rls + phi @ self.P_rls @ phi)
        self.theta = self.theta + K_rls * (y - phi @ self.theta)
        self.P_rls = (1 / self.lambda_rls) * (self.P_rls - np.outer(K_rls, phi @ self.P_rls))
        
        # Update parameters
        self.tau = self.theta[0]
        self.k1 = self.theta[1]
        self.k2 = self.theta[2]
        
        # Update A matrix
        self.A = np.array([[1 - self.dt / self.tau, self.k1 * self.dt / self.tau],
                           [0, 1]])

    def compensate(self, measurements):
        """
        Compensate a sequence of measurements T_meas
        """
        compensated = []
        prev_T = measurements[0]
        for y in measurements:
            Tg_est = self.kalman_step(y)
            self.rls_step(y, Tg_est, self.Te, prev_T)
            compensated.append(Tg_est)
            prev_T = y
        return np.array(compensated)