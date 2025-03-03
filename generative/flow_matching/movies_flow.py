import numpy as np
import tqdm
import math
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# Constants
SIGMA_MIN = 1e-4
TRAINING_STEPS = 50_000
BATCH_SIZE = 64
LEARNING_RATE = 1e-4


from PIL import Image
image = Image.open('generative/flow_matching/bunny1.png') #
# Show the image
#plt.imshow(image)

# GTP Generated Code 
def sample_points_from_image(image_array, num_samples=1000):
    """
    Sample points from an image array based on intensity values.

    Parameters:
    - image_array: numpy array of the image.
    - num_samples: number of points to sample.

    Returns:
    - x, y: coordinates of the sampled points.
    """
    # Normalize image_array to have values between 0 and 1
    normalized_array = 1 - (image_array / 255.0)  # Invert so darker areas have higher values

    # Cumulative distribution function of the normalized values
    cdf = normalized_array.cumsum()
    cdf = cdf / cdf[-1]

    # Generate random numbers and find corresponding indices in the CDF
    random_numbers = np.random.rand(num_samples)
    indices = np.searchsorted(cdf, random_numbers)

    # Convert flat indices back to 2D coordinates
    y, x = np.unravel_index(indices, image_array.shape)

    return x, y

gray_image_rabbit = image.convert('L')
gray_array_rabbit = np.array(gray_image_rabbit)

# Sample points from this new image
x_points_rabbit, y_points_rabbit = sample_points_from_image(gray_array_rabbit, 30000)
# Center the data with spread [-5, 5] using min and max values
x_points_rabbit = (x_points_rabbit - x_points_rabbit.min()) / (x_points_rabbit.max() - x_points_rabbit.min()) * 6 - 3
y_points_rabbit = (y_points_rabbit - y_points_rabbit.min()) / (y_points_rabbit.max() - y_points_rabbit.min()) * 6 - 3
y_points_rabbit = -y_points_rabbit  # Invert y-axis
sampled_points = np.stack([x_points_rabbit, y_points_rabbit], axis=1)
plt.scatter(sampled_points[:,0], sampled_points[:,1], s=1, c='black')
plt.axis('equal')
plt.show()
# Wait for the user to close the plot
plt.show()



class Block(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.ff = nn.Linear(channels, channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.ff(x))

class MLP(nn.Module):
    def __init__(self, channels_data=2, layers=5, channels=512, channels_t=512):
        super().__init__()
        self.channels_t = channels_t
        self.in_projection = nn.Linear(channels_data, channels)
        self.t_projection = nn.Linear(channels_t, channels)
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(layers)])
        self.out_projection = nn.Linear(channels, channels_data)

    def gen_t_embedding(self, t, max_positions=10_000):
        """
        Generates sinusoidal time embeddings.
        t: (batch, 1) tensor of time steps.
        Returns: (batch, channels_t) tensor.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1) # Ensure shape (batch, 1)
        t = t * max_positions
        half_dim = self.channels_t // 2
        emb_scale = math.log(float(max_positions) + 1e-8) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_scale)
        emb = t * emb  # (batch, half_dim)

        # Compute sinusoidal embeddings
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)  # (batch, channels_t)
        
        # If channels_t is odd, pad the embedding with zero
        if self.channels_t % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), mode='constant', value=0)
        
        return emb

    def forward(self, x, t):
        """
        Forward pass of MLP.
        x: (batch, channels_data) input data.
        t: (batch, 1) time step input.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # Ensure shape (batch, 1)

        x = self.in_projection(x)  # (batch, channels)
        t_emb = self.gen_t_embedding(t)  # (batch, channels_t)
        t_emb = self.t_projection(t_emb)  # (batch, channels)

        x = x + t_emb  # Combine time and input embeddings
        x = self.blocks(x)
        x = self.out_projection(x)  # (batch, channels_data)

        return x
    
import torch
import tqdm


# Initialize model and optimizer
model = MLP(layers=5, channels=512)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Load data
data = torch.Tensor(sampled_points)

# Training loop
losses = []
pbar = tqdm.tqdm(range(TRAINING_STEPS), desc="Training Progress")

# Training loop if generative/flow_matching/bunny.pth does not exist
if not os.path.exists('generative/flow_matching/bunny.pth'):
    for _ in pbar:
        # Sample batch
        indices = torch.randint(data.size(0), (BATCH_SIZE,))
        x1 = data[indices]  # Shape: (BATCH_SIZE, data_dim)
        t = torch.rand(BATCH_SIZE, 1, device=x1.device)  # t ~ Uniform(0,1), Shape: (BATCH_SIZE, 1)
        mu = x1 * t  # Shape: (BATCH_SIZE, data_dim)
        sigma = 1 - (1 - SIGMA_MIN) * t  # Shape: (BATCH_SIZE, data_dim)
        x_t = mu + sigma * torch.randn_like(mu)  # x_t ~ N(mu, sigma^2) Shape: (BATCH_SIZE, data_dim)
        u_t = (x1 - (1 - SIGMA_MIN) * x_t) / (1 - (1 - SIGMA_MIN) * t)  # Target velocity
        # Model should predict the Target velocity
        pred = model(x_t, t) # Shape: (BATCH_SIZE, data_dim)
        loss = ((u_t - pred) ** 2).mean()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        pbar.set_postfix(loss=loss.item())
        losses.append(loss.item())

    plt.plot(losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    # Saving Model
    torch.save(model.state_dict(), 'generative/flow_matching/bunny.pth')
else:
    print("Model already trained. Skipping training step.")
    model.load_state_dict(torch.load('generative/flow_matching/bunny.pth'))
    model.eval()

import matplotlib.colors as mcolors
def assign_colors_by_polar_coordinates_with_brightness(points):
    """
    Assigns colors to points based on their polar coordinates.
    - Hue is determined by the angle (θ) from the positive x-axis.
    - Brightness (Value in HSV) is adjusted based on radius (smaller radius → darker).

    Parameters:
    - points: (N, 2) array of 2D points.
    
    Returns:
    - colors: (N, 3) RGB colors assigned based on angle and radius.
    """
    # Compute polar coordinates
    x, y = points[:, 0], points[:, 1]
    angles = np.arctan2(y, x)  # Angle in radians [-pi, pi]
    angles = (angles + np.pi) / (2 * np.pi)  # Normalize to [0,1] (hue)

    radii = np.sqrt(x**2 + y**2)  # Compute radius
    radii = (radii - radii.min()) / (radii.max() - radii.min() + 1e-6)  # Normalize to [0,1]
    brightness = 0.3 + 0.7 * radii  # Scale brightness (avoid too dark values)

    # Convert angle to hue in HSV color space
    hsv_colors = np.array([angles, np.ones_like(angles), brightness]).T  # HSV (hue, saturation, brightness)
    rgb_colors = mcolors.hsv_to_rgb(hsv_colors)  # Convert to RGB

    return rgb_colors

import torchdiffeq 
import matplotlib.animation as animation

# Optionally force computations on CPU (if desired)
torch.set_default_device("cpu")

# Set random seed for reproducibility and switch model to evaluation mode
torch.manual_seed(42)
model.eval()

# Initialize particles and assign colors (using a predefined function)
xt = torch.randn(25000, 2)
colors = assign_colors_by_polar_coordinates_with_brightness(xt.numpy())

steps = 500
plot_every = 1  # Will sample every 'plot_every' steps for the animation
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(xt[:, 0], xt[:, 1], marker="o", c=colors, s=1.5)
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_title("Flow Matching Dynamics (RK4 Integration)")

def velocity_field(t, x_t):
    """
    Computes the velocity at a given time step using the model.
    
    Args:
        t: A scalar time step.
        x_t: Tensor of shape (batch_size, dim).
    
    Returns:
        Tensor of shape (batch_size, dim) representing the velocity.
    """
    # Ensure x_t always has shape (batch_size, dim)
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(0)
    # Create a tensor for t with shape (batch_size, 1)
    t_tensor = torch.full((x_t.shape[0], 1), t.item(), device=x_t.device, dtype=x_t.dtype)
    return model(x_t, t_tensor)

# Define time steps manually to integrate iteratively, reducing memory usage
t_eval = torch.linspace(0, 1, steps, device=xt.device)
trajectories = [xt.clone()]

with torch.no_grad():
    for i in range(1, steps):
        # Integrate one step at a time with RK4
        t_span = torch.tensor([t_eval[i-1], t_eval[i]], device=xt.device)
        xt = torchdiffeq.odeint(velocity_field, xt, t_span, method="rk4")[-1]  # Keep only the last step
        
        if i % plot_every == 0 or i == steps:  # Store only sampled step
            trajectories.append(xt.clone())

def update(frame):
    ax.set_title(f"Step {frame * plot_every}/{steps} - Time: {frame / (len(trajectories) - 1):.2f}")
    sc.set_offsets(trajectories[frame].cpu().numpy())

# Create and save the animation
ani = animation.FuncAnimation(fig, update, frames=len(trajectories), interval=10)
ani.save("flow_matching_animation_bunny.mp4", writer="ffmpeg")

# Restore model to training mode
model.train()

print("Animation saved as flow_matching_animation_bunny.mp4")