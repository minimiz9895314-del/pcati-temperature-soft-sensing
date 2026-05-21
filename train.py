import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from bgct import BayesianGatedConvTransformer
from physics_constraints import physics_loss
from utils import generate_synthetic_data

# Parameters from paper
params = (0.01, 500, 0.001, 50, 30)  # ms, cs, As, hconv, heff
Te = 300
dt = 1.0

# Generate data
Tg, T_wall = generate_synthetic_data(2000)
# Assume internal T is Tg, wall is T_wall
# For training, use sequences
seq_len = 50
X = []
y = []
for i in range(len(T_wall) - seq_len):
    X.append(T_wall[i:i+seq_len])
    y.append(Tg[i+seq_len])
X = np.array(X)[:, :, np.newaxis]  # [N, seq_len, 1]
y = np.array(y)[:, np.newaxis]     # [N, 1]

# Split
split = int(0.75 * len(X))
X_train, X_test = torch.tensor(X[:split]).float(), torch.tensor(X[split:]).float()
y_train, y_test = torch.tensor(y[:split]).float(), torch.tensor(y[split:]).float()

dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = BayesianGatedConvTransformer(input_dim=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.GaussianNLLLoss()

epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        mean, var = model(batch_x.transpose(1, 2))  # [B, 1, seq_len] for conv
        pred_loss = criterion(mean, batch_y, var)
        
        # Physics loss (approx, using last of sequence as prev)
        prev_pred = model(batch_x[:, :-1, :].transpose(1, 2))[0]
        Tg_batch = batch_y  # Approximate Tg with y
        Te_batch = torch.full_like(batch_y, Te)
        phys_loss = physics_loss(mean, prev_pred, Tg_batch, Te_batch, params, dt)
        
        loss = pred_loss + 0.1 * phys_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}: Loss {total_loss / len(loader):.4f}')

# Save model
torch.save(model.state_dict(), 'bgct_model.pth')