import torch
import torch.nn as nn

# 1. Complex Data: Let's pretend this is a pattern a straight line can't solve
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[2.0], [8.0], [18.0], [32.0]]) # Note: This is y = 2 * (x^2) -> a curve!

# 2. Building a Deep Model with nn.Sequential
model = nn.Sequential(
    # Layer 1: Takes 1 input (X) and passes it to 10 hidden neurons
    nn.Linear(in_features=1, out_features=10),
    
    # Activation Function: Adds the ability to learn curves/bends
    nn.ReLU(),
    
    # Layer 2: Takes those 10 neurons and boils them down to 1 output (Y)
    nn.Linear(in_features=10, out_features=1)
)

# 3. Setup our Loss and Optimizer (We use Adam here, a slightly smarter optimizer)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("🧠 Deep Neural Network created with a Hidden Layer and ReLU non-linearity!\n")

# 4. Training Loop (Running for 500 epochs since curves take longer to learn)
for epoch in range(500):
    predictions = model(X)
    loss = loss_fn(predictions, Y)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# 5. Let's test it on a brand new number it has never seen!
# What should 5 equal if the pattern is y = 2 * (x^2)? (5^2 * 2 = 50)
test_input = torch.tensor([[5.0]])

# model(test_input) makes a prediction
# .item() turns the tensor back into a normal Python float
predicted_output = model(test_input).item()

print(f"\n🔮 Testing on X = 5.0")
print(f"Model's Prediction for Y: {predicted_output:.2f} (Target should be close to 50.0!)")
