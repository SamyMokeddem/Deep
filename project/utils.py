import torch
import numpy as np
import torch.nn.functional as F
import dataset as windData
import os
import json
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def save_model(model, run_name, architecture_details):

    directory = "train_models/"
    filename = f"{run_name}.pth"
    path = directory + filename
    json_path = directory + f"{run_name}_architecture_data.json"
    
    os.makedirs(directory, exist_ok=True)

    torch.save(model.state_dict(), path)
    with open(json_path, 'w') as f:
        json.dump(architecture_details, f)
    print(f"Model and metadata saved successfully at {path} and {json_path}")


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded successfully from {path}")
    return model


def resize_high_res(high_res):
    # Convert NumPy array to PyTorch tensor
    high_res_tensor = torch.tensor(high_res).unsqueeze(0)  # Add batch dimension
    # Resize high-resolution data to 64x64
    resized_high_res = F.interpolate(high_res_tensor, size=(64, 64), mode='bilinear', align_corners=False)
    return resized_high_res.squeeze(0)  # Remove batch dimension # .numpy() to turn it back into numpy array

def load_data(var_name=['u10', 'v10'], start=2010, end=2020, pred_step=0, normalize=False, 
              in_max=None, in_min=None, out_max=None, out_min=None):

    in_data, out_data, dates = windData.make_clean_data(var_name, start, end)
    u10 = in_data[:,:,:,0]
    v10 = in_data[:,:,:,1]
    in_data = np.sqrt(np.square(u10) + np.square(v10))
    #normalize the data
    in_norm_raw = (np.max(in_data), np.min(in_data))
    out_norm_raw = (np.max(out_data), np.min(out_data))
    if normalize:
        if in_max is None or in_min is None:
            in_max, in_min = (np.max(in_data), np.min(in_data))
        in_data = (in_data-in_min)/(in_max-in_min)

        if out_max is None or out_min is None:
            out_max, out_min = (np.max(out_data), np.min(out_data))
        out_data = (out_data-out_min)/(out_max-out_min)

    train_dataset = windData.DownscalingDataset(in_data, out_data, dates, low_var_name='uv10', high_var_name='si10', pred_step=pred_step)

    train_dataset.get_var_name()
    print("dataset loaded with a train size of ", len(train_dataset))
    print("The input data has a shape of ", in_data.shape, " and the output data has a shape of ", out_data.shape)
    return train_dataset, in_norm_raw, out_norm_raw


class NoiseScheduler():
    def __init__(self, T, min_noise, max_noise):
        self.T = T
        self.min_noise = min_noise
        self.max_noise = max_noise
        # Define beta schedule
        self.betas = self.linear_beta_schedule(timesteps=T, start=min_noise, end=max_noise)
        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    @staticmethod
    def linear_beta_schedule(timesteps, start, end):
        return torch.linspace(start, end, timesteps)

    @staticmethod
    def get_index_from_list(vals, t, x_shape):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)