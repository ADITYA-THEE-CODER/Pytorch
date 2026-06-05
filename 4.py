import torch
import torch.nn as nn  # <--- NN stands for Neural Networks

# 1. Create data (Same as before)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# 2. Define the Model using PyTorch's built-in layers
# in_features=1 (we give it 1 number, X)
# out_features=1 (it outputs 1 number, Y)
model = nn.Linear(in_features=1, out_features=1)

# 3. Define the Loss Function and the Optimizer
# MSELoss = Mean Squared Error (how we calculate the error)
# SGD = Stochastic Gradient Descent (the optimizer that turns the knobs)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

print(" Model created! PyTorch automatically generated random weights and biases inside it.\n")

# 4. The Training Loop
for epoch in range(100):
    # FORWARD PASS: Pass X into the model to get predictions
    predictions = model(X)
    
    # CALCULATE LOSS: Compare predictions to correct answers (Y)
    loss = loss_fn(predictions, Y)
    
    # BACKWARD PASS: Calculate the gradients automatically
    loss.backward()
    
    # UPDATE KNOBS: Tell the optimizer to nudge the weights and biases
    optimizer.step()
    
    # Clear the gradients for the next round
    optimizer.zero_grad()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# 5. Inspect what the model learned
print("\n  Final learned parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.strip().tolist()}")
