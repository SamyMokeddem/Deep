import os
import datetime
import torch
import torch.nn.functional as F
from datetime import datetime
import torch.optim as optim
from tqdm import tqdm
import math
from utils import save_model, load_model
from evaluate import test_proc
from time import time
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Wandb config
import wandb
os.environ["WANDB_ENTITY"]="WindDownscaling"


def pred_noise(batch, model, ns, device, up_shape=(64, 64)):
    batch_size = batch["high_res"].shape[0]

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

    return noise, noise_pred

def train_proc(model, ns, train_data_loader, val_data_loader, architecture_details, up_shape=(64, 64),
                num_epochs=100, lr=5e-04, run_name=None, save=True, load_best_model=True):
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device) # 200 in paper.
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True, threshold=1e-04)
    run_name = run_name if run_name is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    wandb.init(
        # set the wandb project where this run will be logged
        project="Wind_Downscaling",

        # track hyperparameters and run metadata
        config=architecture_details,
        
        name=run_name
    )

    for epoch in tqdm(range(num_epochs)):
        # callback_lr_wd(optimizer, epoch, num_epochs)
        start_time = time()
        model.train()
        total_loss = 0.0
        total_batches = 0

        best_loss = math.inf

        for i, batch in enumerate(train_data_loader):
            optimizer.zero_grad()

            noise, noise_pred = pred_noise(batch, model, ns, device, up_shape=up_shape)
            
            loss = F.mse_loss(noise, noise_pred) # before: l1_loss

            total_loss += loss.item()
            total_batches += 1

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        average_loss = total_loss/total_batches
        end_time = time()
        wandb.log({
            "train/loss": average_loss, 
            "epoch":epoch, 
            'lr': optimizer.param_groups[0]['lr'],
            'train/time': end_time-start_time})


        # validation part

        model.eval()
        with torch.no_grad(): # not sure needed since we use model.eval() but I guess it cannot harm
            start_time = time()
            val_loss = 0.0
            val_batches = 0

            for i, batch in enumerate(val_data_loader):

                noise, noise_pred = pred_noise(batch, model, ns, device, up_shape=up_shape)

                loss = F.mse_loss(noise, noise_pred) # before: l1_loss
                
                val_loss += loss.item()
                val_batches += 1

            average_val_loss = val_loss/val_batches
            end_time = time()
            #scheduler.step(average_test_loss) # ReduceLR on plateau!
            wandb.log({
                "val/loss": average_val_loss, 
                "epoch":epoch,
                'val/time': end_time-start_time,
                })

        if epoch % 2 == 0:
            start_time = time()
            print(f"Epoch = {epoch+1}/{num_epochs}.")
            print(f"Training Loss over the last epoch = {average_loss}")
            print(f"Validation Loss over the last epoch = {average_val_loss}")
            print(f"Learning rate = {optimizer.param_groups[0]['lr']}") # Trying to see if ReduceLROnPlateau works
            #inference
            # ddim
            train_DDIM_error, train_DDIM_fig = test_proc(model, ns, train_data_loader, num_epoch=epoch, up_shape=up_shape)
            test_DDIM_error, test_DDIM_fig = test_proc(model, ns, val_data_loader, num_epoch=epoch, up_shape=up_shape)
            # ddpm
            train_DDPM_error, train_DDPM_fig = test_proc(model, ns, train_data_loader, num_epoch=epoch, up_shape=up_shape)
            test_DDPM_error, test_DDPM_fig = test_proc(model, ns, val_data_loader, num_epoch=epoch, up_shape=up_shape)
            end_time = time()
            wandb.log({
                "train/DDIM_error": train_DDIM_error,
                "train/DDIM_inference": train_DDIM_fig,
                "val/DDIM_error": test_DDIM_error,
                "val/DDIM_inference": test_DDIM_fig,
                "train/DDPM_error": train_DDPM_error,
                "train/DDPM_inference": train_DDPM_fig,
                "val/DDPM_error": test_DDPM_error,
                "val/DDPM_inference": test_DDPM_fig,
                "epoch":epoch,
                'inference/time': end_time-start_time,
                })

        if load_best_model:
            if average_val_loss < best_loss:
                best_loss = average_val_loss
                save_model(model, run_name+"_best", architecture_details=architecture_details)

    if save:
        save_model(model, run_name+"_final", architecture_details=architecture_details)
    
    if load_best_model:
        model= load_model(model, "train_models/" + run_name + "_best.pth")

    wandb.finish()