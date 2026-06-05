import torch

# 1. Create our data (X is input, Y is the correct output)
# We use the formula y = 2x + 1 to make this data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

# 2. Initialize a random Weight (w) and Bias (b)
# This is the computer's starting "guess". 
# requires_grad=True tells PyTorch: "Track this variable, we need to optimize it!"
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f" Starting random guess -> Weight: {w.item():.2f}, Bias: {b.item():.2f}\n")

learning_rate = 0.05

# 3. The Training Loop (The Guess-and-Adjust game)
for epoch in range(200):
    
    # FORWARD PASS: Make a prediction using current w and b (y = w*x + b)
    predictions = X @ w + b
    
    # CALCULATE LOSS: How far off are the guesses? (Mean Squared Error)
    loss = torch.mean((predictions - Y) ** 2)
    
    # BACKWARD PASS: PyTorch does the automatic calculus here!
    # It calculates exactly how to change w and b to make the loss smaller.
    loss.backward()
    
    # UPDATE THE KNOBS: Adjust w and b slightly based on the calculus
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # Zero out the gradients so they don't accumulate for the next round
        w.grad.zero_()
        b.grad.zero_()
        
    # Print progress every 20 steps
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f} -> Weight: {w.item():.2f}, Bias: {b.item():.2f}")

print(f"\n Final Results -> Predicted Weight: {w.item():.2f}, Predicted Bias: {b.item():.2f}")
