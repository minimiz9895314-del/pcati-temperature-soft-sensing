# PCATI: Physics-Constrained Adaptive Thermal Intelligence Framework

This repository implements the PCATI framework from the paper "Physics-Constrained Bayesian Transformer for Temperature Soft Sensing with Uncertainty Quantification in Nuclear Waste Incinerators" by Bo Zhang et al.

## Overview
- **DTC**: Dynamic Temperature Compensation using Kalman filter and Recursive Least Squares (RLS).
- **BGCT**: Bayesian Gated Convolutional Transformer for temperature prediction with uncertainty quantification.
- **Physics-Constrained Training**: Integrates thermodynamic constraints into the loss function.

## Installation
1. Clone the repo:
2. Install dependencies:


This release introduces the revised PCATI framework codebase accompanying the manuscript revision. Key updates include:

- DTC module: fixed ambient temperature input handling in the Kalman prediction step, added support for time-varying ambient temperature sequences, and documented the RLS forgetting factor selection basis 
  (grid search over {0.95, 0.96, 0.97, 0.98, 0.99} on the validation combustion cycle).

- BGCT module: fixed syntax error in GatedConvModule, resolved missing numpy import in TemperatureAwareAttention, and added predict_with_uncertainty() method that returns decomposed epistemic 
  and aleatoric uncertainty components separately. Documented dropout rate and architecture hyperparameter selection via grid search.

- Physics constraints module: fixed duplicate steady_state_loss definition, fixed missing As parameter in steady_state_loss, added AdaptiveLossWeights class encapsulating the adaptive lambda scheduling 
  from Section 3.3, and added nll_loss function. Documented initial loss weight selection basis (lambda1=1.0, lambda2=0.1, lambda3=0.01).

- README: added hyperparameter selection tables covering DTC, BGCT, and physics-constrained training, addressing reviewer reproducibility requirements.
