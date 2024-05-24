import os
import json
import warnings

import numpy as np
import torch
import torch.nn.functional as F

import dataset as windData

# Ignore warnings
warnings.filterwarnings("ignore")

def save_model(model, run_name, architecture_details):
    """Saves the model and its architecture details."""
    directory = "train_models/"
    path = os.path.join(directory, f"{run_name}.pth")
    json_path = os.path.join(directory, f"{run_name}_architecture_data.json")
    
    os.makedirs(directory, exist_ok=True)

    torch.save(model.state_dict(), path)
    with open(json_path, 'w') as f:
        json.dump(architecture_details, f)
    
    print(f"Model and metadata saved successfully at {path} and {json_path}")

def load_model(model, path):
    """Loads the model from the given path."""
    model.load_state_dict(torch.load(path))
    print(f"Model loaded successfully from {path}")
    return model

def load_data(var_name=['u10', 'v10'], start=2010, end=2020, pred_step=0, normalize=False, 
              in_max=None, in_min=None, out_max=None, out_min=None):
    """Loads and processes the data for training."""
    in_data, out_data, dates = windData.make_clean_data(var_name, start, end)
    
    # Compute wind speed magnitude from u10 and v10 components
    u10 = in_data[:, :, :, 0]
    v10 = in_data[:, :, :, 1]
    in_data = np.sqrt(np.square(u10) + np.square(v10))
    
    in_norm_raw = (np.max(in_data), np.min(in_data))
    out_norm_raw = (np.max(out_data), np.min(out_data))
    
    # Normalize the data if required
    if normalize:
        if in_max is None or in_min is None:
            in_max, in_min = in_norm_raw
        in_data = (in_data - in_min) / (in_max - in_min)

        if out_max is None or out_min is None:
            out_max, out_min = out_norm_raw
        out_data = (out_data - out_min) / (out_max - out_min)

    train_dataset = windData.DownscalingDataset(in_data, out_data, dates, low_var_name='uv10', high_var_name='si10', pred_step=pred_step)

    train_dataset.get_var_name()
    
    print(f"Dataset loaded with a train size of {len(train_dataset)}")
    print(f"The input data has a shape of {in_data.shape} and the output data has a shape of {out_data.shape}")
    
    return train_dataset, in_norm_raw, out_norm_raw

class NoiseScheduler:
    def __init__(self, T, min_noise, max_noise):
        """Initializes the NoiseScheduler with the given parameters."""
        self.T = T
        self.min_noise = min_noise
        self.max_noise = max_noise

        # Define beta schedule
        self.betas = torch.linspace(min_noise, max_noise, T)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, axis=0)
        self.sqrt_inv_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)

    @staticmethod
    def get_index_from_list(vals, t, x_shape):
        """Returns a specific index t of a passed list of values vals while considering the batch dimension."""
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def add_noise(self, x_0, t, device="cpu"):
        """Adds noise to the input data x_0 at timestep t."""
        noise = torch.randn_like(x_0)
        sqrt_alphas_bar_t = self.get_index_from_list(self.sqrt_alphas_bar, t, x_0.shape)
        sqrt_one_minus_alphas_bar_t = self.get_index_from_list(self.sqrt_one_minus_alphas_bar, t, x_0.shape)

        return sqrt_alphas_bar_t.to(device) * x_0.to(device) \
               + sqrt_one_minus_alphas_bar_t.to(device) * noise.to(device), noise.to(device)