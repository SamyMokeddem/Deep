import os
import datetime
import torch
import torch.nn.functional as F
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm
import math
from utils import save_model, load_model
from evaluate import eval_proc
from time import time
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Wandb config
import wandb
os.environ["WANDB_ENTITY"]="WindDownscaling"

def pred_noise(batch, model, ns, device, up_shape=(64, 64)): # Function that adds noise to given batch and then predicts the noise that was added
    batch_size = batch["high_res"].shape[0] # Extract batch size

    high_res_imgs = batch["high_res"] # Extract high resolution images
    high_res_imgs = high_res_imgs.unsqueeze(1).float() # We unsqueeze to get the channel dimension
    upsamp_hr = F.interpolate(high_res_imgs, size=up_shape, mode='bilinear')
    high_res_imgs = upsamp_hr

    # We adjust in case the last batch does not contain batch_size elements
    if high_res_imgs.shape[0] != batch_size:
        t = torch.randint(0, ns.T, (high_res_imgs.shape[0],), device=device).long()
    else :
        t = torch.randint(0, ns.T, (batch_size,), device=device).long()

    # Apply noise
    noisy_x, noise = ns.add_noise(high_res_imgs, t)
    noisy_x = noisy_x.float()
    noise = noise.to(device)

    low_res_imgs = batch["low_res"] # Extract low resolution images
    low_res_imgs = low_res_imgs.float()

    # Upscale the low resolution images so that they have the same shape as the high resolution images
    upsampled = F.interpolate(low_res_imgs, size=up_shape, mode='bilinear')

    # Concatenate them along the channel dimension
    unet_input = torch.cat((noisy_x, upsampled), dim=1)

    # Predict the noise that was added
    noise_pred = model(unet_input.to(device), t)

    return noise, noise_pred

 # Function that implements the training loop
def train_proc(model, ns, train_data_loader, val_data_loader, architecture_details, up_shape=(64, 64),
                num_epochs=100, lr=5e-04, ensemble=1, run_name=None, save=True, load_best_model=True):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    run_name = run_name if run_name is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    wandb.init(
        # Set the wandb project where this run will be logged
        project="Report",

        # Track hyperparameters and run metadata
        config=architecture_details,
        
        name=run_name
    )

    for epoch in tqdm(range(num_epochs)):
        start_time = time()
        model.train()
        total_loss = 0.0 # Total loss over the epoch
        total_batches = 0 # Total number of batches over the epoch

        best_loss = math.inf

        for i, batch in enumerate(train_data_loader): # Training loop
            optimizer.zero_grad()

            noise, noise_pred = pred_noise(batch, model, ns, device, up_shape=up_shape)
            
            loss = F.mse_loss(noise, noise_pred)

            total_loss += loss.item()
            total_batches += 1

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping to avoid exploding gradients

            optimizer.step()

        average_loss = total_loss/total_batches # Evaluate the loss over the epoch
        end_time = time()
        wandb.log({
            "train/loss": average_loss, 
            "epoch":epoch, 
            'lr': optimizer.param_groups[0]['lr'],
            'train/time': end_time-start_time})


        # Validation of the model
        model.eval()
        with torch.no_grad():
            start_time = time()
            val_loss = 0.0
            val_batches = 0

            for i, batch in enumerate(val_data_loader):

                noise, noise_pred = pred_noise(batch, model, ns, device, up_shape=up_shape)

                loss = F.mse_loss(noise, noise_pred)
                
                val_loss += loss.item()
                val_batches += 1

            average_val_loss = val_loss/val_batches
            end_time = time()
            wandb.log({
                "val/loss": average_val_loss, 
                "epoch":epoch,
                'val/time': end_time-start_time,
                })

        start_time = time()
        print(f"Epoch = {epoch+1}/{num_epochs}.")
        print(f"Training Loss over the last epoch = {average_loss}")
        print(f"Validation Loss over the last epoch = {average_val_loss}")

        # We evaluate the current inference results both on the training set and validation set
        print("Starting evaluation...")
        train_DDPM_mse, train_DDPM_ssim, train_DDPM_fig = eval_proc(model, ns, train_data_loader, num_epoch=epoch, up_shape=up_shape, ensemble=ensemble)
        test_DDPM_mse, test_DDPM_ssim, test_DDPM_fig = eval_proc(model, ns, val_data_loader, num_epoch=epoch, up_shape=up_shape, ensemble=ensemble)
        end_time = time()
        wandb.log({
            "train/DDPM_error": train_DDPM_mse,
            "val/DDPM_error": test_DDPM_mse,
            "train/DDPM_ssim": train_DDPM_ssim,
            "val/DDPM_ssim": test_DDPM_ssim,
            "train/DDPM_inference": train_DDPM_fig,
            "val/DDPM_inference": test_DDPM_fig,
            "epoch":epoch,
            'inference/time': end_time-start_time,
            })
        print("End of evaluation...")
        # If we are saving the model, we save the best model so far
        if load_best_model:
            print("Best loss so far: ", best_loss)
            if average_val_loss < best_loss:
                best_loss = average_val_loss
                save_model(model, run_name+"_best", architecture_details=architecture_details)

    if save:
        save_model(model, run_name+"_final", architecture_details=architecture_details)
    
    if load_best_model:
        model= load_model(model, "train_models/" + run_name + "_best.pth")

    wandb.finish()