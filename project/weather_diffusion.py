import os
import datetime
import json
import torch
import numpy as np
import netCDF4 as nc
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from datetime import datetime
import torch.optim as optim
import dataset as windData
from tqdm import tqdm
from models import DenoisingUnet, SimpleUnet
import matplotlib.pyplot as plt
import math
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Wandb config
import wandb
os.environ["WANDB_ENTITY"]="WindDownscaling"

def resize_high_res(high_res):
    # Convert NumPy array to PyTorch tensor
    high_res_tensor = torch.tensor(high_res).unsqueeze(0)  # Add batch dimension
    # Resize high-resolution data to 64x64
    resized_high_res = F.interpolate(high_res_tensor, size=(64, 64), mode='bilinear', align_corners=False)
    return resized_high_res.squeeze(0)  # Remove batch dimension # .numpy() to turn it back into numpy array

def load_data(var_name=['u10', 'v10'], start=2010, end=2020, split_ratio = [0.2, 0.2]):

    in_data, out_data = windData.make_clean_data(var_name, start, end)
    u10 = in_data[:,:,:,0]
    v10 = in_data[:,:,:,1]
    in_data = np.sqrt(np.square(u10) + np.square(v10))

    #normalize the data
    in_max, in_min = (np.max(in_data), np.min(in_data))
    in_data = (in_data-in_max)/(in_max-in_min)

    out_max, out_min = (np.max(out_data), np.min(out_data))
    out_data = (out_data-out_max)/(out_max-out_min)

    train_idx, val_idx, test_idx = windData.spliting_indices(len(in_data), val_pct=split_ratio[0], test_pct=split_ratio[1])

    # Diviser le dataset en trois pour que ce soit propre.

    train_dataset = windData.DownscalingDataset(in_data, out_data, low_var_name='uv10', high_var_name='si10', indices=train_idx)
    test_dataset = windData.DownscalingDataset(in_data, out_data, low_var_name='uv10', high_var_name='si10', indices=val_idx)
    val_dataset = windData.DownscalingDataset(in_data, out_data, low_var_name='uv10', high_var_name='si10', indices= test_idx)

    train_dataset.get_var_name()
    print("dataset loaded with a train size of ", len(train_dataset), " and a test size of ", len(test_dataset))
    print("The input data has a shape of ", in_data.shape, " and the output data has a shape of ", out_data.shape)
    return train_dataset, test_dataset, val_dataset


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


def train_proc(model, ns, train_data_loader, val_data_loader, architecture_details, up_shape=(64, 64), num_epochs=100, batch_size=32, lr=5e-04, run_name=None, save=True, load_best_model=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device) # 200 in paper.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load the data
    

    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True, threshold=1e-04)
    run_name = run_name if run_name is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    wandb.init(
        # set the wandb project where this run will be logged
        project="Wind_Downscaling",

        # track hyperparameters and run metadata
        config={
        "architecture": "UNet-diffuse",
        "channels": "T",
        "dataset": "2010-2020",
        "normalize": "True",
        "learning_rate": lr,
        "epochs": num_epochs
        },
        
        name=run_name
    )

    for epoch in tqdm(range(num_epochs)):
        # callback_lr_wd(optimizer, epoch, num_epochs)

        model.train()
        total_loss = 0.0
        total_batches = 0

        best_loss = math.inf

        for i, batch in enumerate(train_data_loader):
            optimizer.zero_grad()

            high_res_imgs = batch["high_res"]
            high_res_imgs = high_res_imgs.unsqueeze(1).float() # because one channel
            
            # !!! Now also upsample high res so that dimensions (64, 64) ==> downblocks dimensions match upblocks dimensions
            upsamp_hr = F.interpolate(high_res_imgs, size=up_shape, mode='bilinear')
            high_res_imgs = upsamp_hr

            if high_res_imgs.shape[0] != batch_size:
                t = torch.randint(0, ns.T, (high_res_imgs.shape[0],), device=device).long()
            else :
                t = torch.randint(0, ns.T, (batch_size,), device=device).long() # En gros, si le dernier batch est plus petit faut faire ca pour que le input de forward difusion soit de même taille

            # apply noise
            noisy_x, noise = ns.forward_diffusion_sample(high_res_imgs, t)

            noisy_x = noisy_x.float()
            
            # For now, let's only condition on low res at the same time, to simplify implementation
            low_res_imgs = batch["low_res"]
            low_res_imgs = low_res_imgs.unsqueeze(1).float() # because one channel

            # upsample low res to match high res shape
            upsampled = F.interpolate(low_res_imgs, size=up_shape, mode='bilinear')

            unet_input = torch.cat((noisy_x, upsampled), dim=1)

            noise_pred = model(unet_input.to(device), t) # model is denoising unet
            
            noise = noise.to(device)
            loss = F.mse_loss(noise, noise_pred) # before: l1_loss

            total_loss += loss.item()
            total_batches += 1

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        average_loss = total_loss/total_batches
        wandb.log({"loss": average_loss, "epoch":epoch, 'lr': optimizer.param_groups[0]['lr']})


        # validation part

        model.eval()
        with torch.no_grad(): # not sure needed since we use model.eval() but I guess it cannot harm
            val_loss = 0.0
            val_batches = 0

            for i, batch in enumerate(val_data_loader):

                high_res_imgs = batch["high_res"]
                high_res_imgs = high_res_imgs.unsqueeze(1).float() # because one channel
                
                # !!! Now also upsample high res so that dimensions (64, 64) ==> downblocks dimensions match upblocks dimensions
                upsamp_hr = F.interpolate(high_res_imgs, size=up_shape, mode='bilinear')
                high_res_imgs = upsamp_hr

                if high_res_imgs.shape[0] != batch_size:
                    t = torch.randint(0, ns.T, (high_res_imgs.shape[0],), device=device).long()
                else :
                    t = torch.randint(0, ns.T, (batch_size,), device=device).long()

                # apply noise
                noisy_x, noise = ns.forward_diffusion_sample(high_res_imgs, t)

                noisy_x = noisy_x.float()

                # For now, let's only condition on low res at the same time, to simplify implementation
                low_res_imgs = batch["low_res"]
                low_res_imgs = low_res_imgs.unsqueeze(1).float() # because one channel

                # upsample low res to match high res shape
                upsampled = F.interpolate(low_res_imgs, size=up_shape, mode='bilinear')

                unet_input = torch.cat((noisy_x, upsampled), dim=1)

                noise_pred = model(unet_input.to(device), t) # model is denoising unet

                noise = noise.to(device)
                loss = F.mse_loss(noise, noise_pred) # before: l1_loss
                
                val_loss += loss.item()
                val_batches += 1

            average_val_loss = val_loss/val_batches
            #scheduler.step(average_test_loss) # ReduceLR on plateau!
            wandb.log({"test_loss": average_val_loss, "epoch":epoch})

        if epoch % 5 == 0:
            print(f"Epoch = {epoch+1}/{num_epochs}.")
            print(f"Training Loss over the last epoch = {average_loss}")
            print(f"Validation Loss over the last epoch = {average_val_loss}")
            print(f"Learning rate = {optimizer.param_groups[0]['lr']}") # Trying to see if ReduceLROnPlateau works
            #inference
            # ddim
            test_proc(model, ns, train_data_loader, train=True, ddpm=False, num_epoch=epoch, up_shape=up_shape)
            test_proc(model, ns, val_data_loader, train=False, ddpm=False, num_epoch=epoch, up_shape=up_shape)
            # ddpm
            test_proc(model, ns, train_data_loader, train=True, ddpm=True, num_epoch=epoch, up_shape=up_shape)
            test_proc(model, ns, val_data_loader, train=False, ddpm=True, num_epoch=epoch, up_shape=up_shape)


        if load_best_model:
            if average_val_loss < best_loss:
                best_loss = average_val_loss
                save_model(model, run_name+"_best", architecture_details=architecture_details)

    if save:
        save_model(model, run_name+"_final", architecture_details=architecture_details)
    
    if load_best_model:
        model= load_model(model, "train_models/" + run_name + "_best.pth")

    wandb.finish()
    
def test_proc(model, ns, data_loader, train=False, ddpm=False, num_epoch=200, up_shape=(64,64)):
    batch = next(iter(data_loader))
    batch_size = 3 # Simpler plot + faster computations, can change that back later
    batch["low_res"] = batch["low_res"][:batch_size]
    batch['high_res'] = batch['high_res'][:batch_size]

    if ddpm:
        all_x_t = DDPM_infer(model, ns, batch, up_shape)
    else:
        all_x_t = DDIM_infer(model, ns, batch, up_shape)

    fig, axs = plt.subplots(len(all_x_t), batch_size, figsize=(6, 12))

    fig.suptitle(f"Evolution of prediction over time steps at Epoch {num_epoch}")

    for i in range(len(all_x_t)):
        for j in range(batch_size):
            axs[i, j].imshow(all_x_t[i][j].squeeze().to('cpu').numpy())

    if ddpm:
        if train:
            wandb.log({"ddpm_train": fig})
            # plt.savefig(f'inference/ddpm_prediction_train_{num_epoch}.png')
        else:
            wandb.log({"ddpm_val": fig})
            # plt.savefig(f'inference/ddpm_prediction_val_{num_epoch}.png')
    else:
        if train:
            wandb.log({"ddim_train": fig})
            # plt.savefig(f'inference/ddim_prediction_train_{num_epoch}.png')
        else:
            wandb.log({"ddim_val": fig})
            # plt.savefig(f'inference/ddim_prediction_val_{num_epoch}.png')
    plt.close()

def DDPM_infer(model, ns, batch, up_shape):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = batch["low_res"].shape[0]
    low_res_imgs = batch["low_res"].unsqueeze(1)
    upsampled = F.interpolate(low_res_imgs, size=up_shape, mode='bilinear').to(device)

    model.to(device)

    n_steps = 5

    all_x_t = []

    x_t = torch.randn((batch_size, 1, up_shape[0], up_shape[1])).to(device)
    all_x_t.append(x_t)
    for i in range(ns.T-1, -1, -1):
        t = torch.tensor([i for _ in range(batch_size)]).to(device).long()
        
        unet_input = torch.cat((x_t, upsampled), dim=1)
        unet_input = unet_input.float().to(device)

        betas_t = ns.get_index_from_list(ns.betas, t, x_t.shape).to(device)

        sqrt_one_minus_alphas_cumprod_t = ns.get_index_from_list(ns.sqrt_one_minus_alphas_cumprod, t, x_t.shape).to(device)

        sqrt_recip_alphas_t = ns.get_index_from_list(ns.sqrt_recip_alphas, t, x_t.shape).to(device)
        
        with torch.no_grad():

            pred_noise = model(unet_input, t).to(device)
            x_t_minus_1 = sqrt_recip_alphas_t * (x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)

            posterior_variance_t = ns.get_index_from_list(ns.posterior_variance, t, x_t.shape).to(device)

            if i == 0:
                x_t = x_t_minus_1
            else:
                noise = torch.randn_like(x_t).to(device)
                x_t = x_t_minus_1 + torch.sqrt(posterior_variance_t) * noise
                
            if i % (ns.T/n_steps) == 0:
                all_x_t.append(x_t)
    
    return all_x_t

def DDIM_infer(model, ns, batch, up_shape):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = batch["low_res"].shape[0]
    low_res_imgs = batch["low_res"].unsqueeze(1)
    upsampled = F.interpolate(low_res_imgs, size=up_shape, mode='bilinear').to(device)

    model.to(device)

    n_steps = 5

    all_x_t = []

    x_t = torch.randn((batch_size, 1, up_shape[0], up_shape[1])).to(device)
    all_x_t.append(x_t)
    for i in range(ns.T-1, -1, -1):
        t = torch.tensor([i for _ in range(batch_size)]).to(device).long()
        
        unet_input = torch.cat((x_t, upsampled), dim=1)
        unet_input = unet_input.float().to(device)

        recip_sqrt_alphas_cumprod_t = ns.get_index_from_list((1.0 / ns.sqrt_alphas_cumprod), t, x_t.shape).to(device)

        sqrt_one_minus_alphas_cumprod_t = ns.get_index_from_list(ns.sqrt_one_minus_alphas_cumprod, t, x_t.shape).to(device)
        

        with torch.no_grad():
            pred_noise = model(unet_input, t)

            predicted_x0 = recip_sqrt_alphas_cumprod_t * (x_t - sqrt_one_minus_alphas_cumprod_t * pred_noise)
            predicted_x0 = predicted_x0.to(device)
            
            if i == 0:
                x_t = predicted_x0
            else:
                t_minus_1 = t - 1
                t_minus_1 = t_minus_1.to(device)
        
                sqrt_alphas_cumprod_t_minus_1 = ns.get_index_from_list(ns.sqrt_alphas_cumprod, t_minus_1, x_t.shape).to(device)
        
                sqrt_one_minus_alphas_cumprod_t_minus_1 = ns.get_index_from_list(ns.sqrt_one_minus_alphas_cumprod, t_minus_1, x_t.shape).to(device)
                x_t = sqrt_alphas_cumprod_t_minus_1 * predicted_x0 + sqrt_one_minus_alphas_cumprod_t_minus_1 * pred_noise
                
            if i % (ns.T/n_steps) == 0:
                all_x_t.append(x_t)
    
    return all_x_t

if __name__ == "__main__":
    train = True
    test = False

    num_epochs = 2000

    print("loading of the data")
    train_dataset, test_dataset, val_dataset = load_data(['u10', 'v10'], 2010, 2019, split_ratio = [0.1, 0.1])

    batch_size = 128

    train_data_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle=True)

    print("Data loaded")
    # Define the model
    print("Model creation")
    unet_channels = (64, 128, 256, 512, 1024) # <-- From youtube video. In the paper: [64, 128, 256, 384]
    kernel_sizes = [3, 3, 3, 3]
    input_channels = 2
    output_channels = 1
    time_emb_dim = 32 # I guess that's a standard value
    dropout = 0.1
    lr = 1e-04
    up_shape = (64, 64)

    model_type = "simple" # "denoising" or "simple"
    if model_type == "denoising":
        model = DenoisingUnet(
            unet_channels, 
            input_channels, 
            output_channels, 
            kernel_sizes,
            time_emb_dim, 
            up_shape, 
            dropout=dropout, 
            attention=True,
            )
    else:
        model = SimpleUnet(
            unet_channels,
            input_channels,
            output_channels,
            time_emb_dim
        )

    min_noise = 0.0001
    max_noise = 0.02
    T = 300
    ns = NoiseScheduler(T, min_noise, max_noise)
    print("model created")
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    architecture_details = {
        # I removed these two because they are not JSON serializable
        # "Model": model, 
        # "Noise Scheduler": ns,
        "Min noise": min_noise,
        "Max noise": max_noise,
        "T": T,
        "Epochs": num_epochs,
        "Batch size": batch_size,
        "lr": lr,
        "dropout": dropout,
        "unet_channels": unet_channels,
        "kernel_sizes": kernel_sizes,
        "input_channels": input_channels,
        "time_emb_dim": time_emb_dim
        }

    if train:
        print("start of the training process")
        train_proc(
            model, 
            ns, 
            train_data_loader, 
            val_data_loader,
            architecture_details,
            up_shape=up_shape,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            # run_name="SimpleUnet", # <-- Change that to the name of the model
            save=True,
            load_best_model=True
            )

        test_proc(model, ns, test_data_loader, num_epoch=num_epochs)
        print("end of the training process")

    if test:
        print("testing of the saved model")
        model = load_model(model, 'train_models/2024-05-02_17-31-05_best.pth')
        test_proc(model, ns, test_data_loader, num_epoch=num_epochs)
        print("end of the testing process")