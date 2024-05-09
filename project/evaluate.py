import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Wandb config
import wandb
os.environ["WANDB_ENTITY"]="WindDownscaling"


def test_proc(model, ns, data_loader, type='DDPM', num_epoch=200, up_shape=(64,64)):
    batch = next(iter(data_loader)) # We only evaluate 1 batch. Doing all the loader would be more accurate but much slower
    # (could change that later)

    high_res_imgs = batch["high_res"]
    high_res_imgs = high_res_imgs.unsqueeze(1).float() # because one channel

    # !!! Now also upsample high res so that dimensions (64, 64) ==> downblocks dimensions match upblocks dimensions
    upsamp_hr = F.interpolate(high_res_imgs, size=up_shape, mode='bilinear')

    all_x_t, error, fig = None, None, None
    if type == 'DDPM':
        all_x_t, error = DDPM_infer(model, ns, batch, up_shape)
    elif type == 'DDIM':
        all_x_t, error = DDIM_infer(model, ns, batch, up_shape)
    else:
        raise ValueError("type must be 'DDPM' or 'DDIM'")
    
    if all_x_t:
        n_plots = 3 # We only plot the n_plots first images of the batch

        fig, axs = plt.subplots(len(all_x_t) + 1, n_plots, figsize=(9, 12), gridspec_kw={'hspace': 0.5})

        fig.suptitle(f"Evolution of prediction over time steps at Epoch {num_epoch}")

        for i in range(len(all_x_t)):
            for j in range(n_plots):
                axs[i, j].imshow(all_x_t[i][j].squeeze().to('cpu').numpy())
                if i == 0:
                    axs[i, j].set_title("Prediction")

        for j in range(n_plots):
            axs[-1, j].set_title("True high resolution")
            axs[-1, j].imshow(upsamp_hr[j].squeeze().to('cpu').numpy())
        
    return error, fig


def DDPM_infer(model, ns, batch, up_shape):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = batch["low_res"].shape[0]
    low_res_imgs = batch["low_res"].unsqueeze(1)
    upsampled = F.interpolate(low_res_imgs, size=up_shape, mode='bilinear').to(device)
    
    high_res_imgs = batch["high_res"]
    high_res_imgs = high_res_imgs.unsqueeze(1).float() # because one channel

    # !!! Now also upsample high res so that dimensions (64, 64) ==> downblocks dimensions match upblocks dimensions
    upsamp_hr = F.interpolate(high_res_imgs, size=up_shape, mode='bilinear').to(device)

    model.to(device)

    n_steps = 4

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

    avg_error = F.mse_loss(x_t, upsamp_hr) # MSE? L1?
    return all_x_t, avg_error

def DDIM_infer(model, ns, batch, up_shape):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = batch["low_res"].shape[0]
    low_res_imgs = batch["low_res"].unsqueeze(1)
    upsampled = F.interpolate(low_res_imgs, size=up_shape, mode='bilinear').to(device)
    
    high_res_imgs = batch["high_res"]
    high_res_imgs = high_res_imgs.unsqueeze(1).float() # because one channel

    # !!! Now also upsample high res so that dimensions (64, 64) ==> downblocks dimensions match upblocks dimensions
    upsamp_hr = F.interpolate(high_res_imgs, size=up_shape, mode='bilinear').to(device)

    model.to(device)

    n_steps = 4

    all_x_t = []

    x_t = torch.randn((batch_size, 1, up_shape[0], up_shape[1])).to(device)
    all_x_t.append(x_t)
    for i in range(ns.T-1, -1, -1):
        t = torch.tensor([i for _ in range(batch_size)]).to(device).long()
        
        unet_input = torch.cat((x_t, upsampled), dim=1)
        unet_input = unet_input.float().to(device)

        sqrt_alphas_cumprod_t = ns.get_index_from_list(ns.sqrt_alphas_cumprod, t, x_t.shape).to(device)

        sqrt_one_minus_alphas_cumprod_t = ns.get_index_from_list(ns.sqrt_one_minus_alphas_cumprod, t, x_t.shape).to(device)
        

        with torch.no_grad():
            pred_noise = model(unet_input, t)

            predicted_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t
            predicted_x0 = predicted_x0.to(device)
            
            if i == 0:
                x_t = predicted_x0
            else:
                t_minus_1 = torch.tensor([(i-1) for _ in range(batch_size)]).to(device).long()
        
                sqrt_alphas_cumprod_t_minus_1 = ns.get_index_from_list(ns.sqrt_alphas_cumprod, t_minus_1, x_t.shape).to(device)
        
                sqrt_one_minus_alphas_cumprod_t_minus_1 = ns.get_index_from_list(ns.sqrt_one_minus_alphas_cumprod, t_minus_1, x_t.shape).to(device)
                x_t = sqrt_alphas_cumprod_t_minus_1 * predicted_x0 + sqrt_one_minus_alphas_cumprod_t_minus_1 * pred_noise
                
            if i % (ns.T/n_steps) == 0:
                all_x_t.append(x_t)
    
    avg_error = F.mse_loss(x_t, upsamp_hr) # MSE? L1

    return all_x_t, avg_error