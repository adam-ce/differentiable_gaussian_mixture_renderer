 
    
import torch
import torch.nn.functional as fun
import matplotlib.pyplot as plt
import numpy as np

#def inverse_softplus(x):
#    return x + torch.log(-torch.expm1(-x))

def inverse_softplus(y, beta):
    return (1 / beta) * torch.log(torch.exp(beta * y) - 1)

# Generate input data
x = torch.linspace(-10, 10, 400)

# Define the beta values
beta_values = [1, 2, 3, 4, 5]
beta_values = [1, 2, 5, 10]

# Create the plot
plt.figure(figsize=(2, 3))

# Plot the softplus function for each beta value
for beta in beta_values:
    y = torch.exp(-fun.softplus(x, beta))
    plt.plot(x.numpy(), y.numpy(), label=f'exp(-Softplus(x, beta={beta}))')
    # y = fun.softplus(x, beta)
    # plt.plot(x.numpy(), y.numpy(), label=f'Softplus(x, beta={beta})')
    #y = inverse_softplus(x, beta)
    #plt.plot(x.numpy(), y.numpy(), label=f'der beta={beta}')

y = (1 - fun.sigmoid(x))
plt.plot(x.numpy(), y.numpy(), label=f'1 - sigmoid(x)')
# Add title and labels
plt.title('Different activation functions')
plt.xlabel('Input')
plt.ylabel('Transparency')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
