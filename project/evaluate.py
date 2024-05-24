import os
import json
import warnings
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm

# Wandb config
import wandb
os.environ["WANDB_ENTITY"] = "WindDownscaling"

# Ignore warnings
warnings.filterwarnings("ignore")

# Matplotlib configuration for LaTeX-style plots
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


def denormalizing(data, max_val, min_val):
    return data * (max_val - min_val) + min_val


def eval_proc(model, ns, data_loader, num_epoch=200, up_shape=(64, 64), ensemble=1, architecture_details=None, denormalize=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = next(iter(data_loader))
    high_res_imgs = batch["high_res"].unsqueeze(1).float()  # because one channel
    upsamp_hr = F.interpolate(high_res_imgs, size=up_shape, mode='bilinear').to(device)

    ensemble_all_x_t = [torch.stack(DDPM_infer(model, ns, batch, up_shape)) for _ in range(ensemble)]
    all_x_t = torch.stack(ensemble_all_x_t).mean(dim=0)
    x_t = all_x_t[-1]

    if denormalize:
        if architecture_details is None:
            raise ValueError("Architecture details must be provided to denormalize the data")
        x_t = denormalizing(x_t, float(architecture_details['Max_norm_high']), float(architecture_details['Min_norm_high']))
        upsamp_hr = denormalizing(upsamp_hr, float(architecture_details['Max_norm_high']), float(architecture_details['Min_norm_high']))
        data_range = float(architecture_details['Max_norm_high']) - float(architecture_details['Min_norm_high'])
        ssim_measure = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
    else:
        ssim_measure = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    mse = F.mse_loss(x_t, upsamp_hr)
    ssim = ssim_measure(x_t, upsamp_hr)

    n_plots = 3
    fig, axs = plt.subplots(len(all_x_t) + 1, n_plots, figsize=(9, 12), gridspec_kw={'hspace': 0.5})
    fig.suptitle(f"Evolution of prediction over time steps at Epoch {num_epoch}")

    for i in range(len(all_x_t)):
        for j in range(n_plots):
            axs[i, j].imshow(np.flipud(all_x_t[i][j].squeeze().to('cpu').numpy()))
            if i == 0:
                axs[i, j].set_title("Prediction")

    for j in range(n_plots):
        axs[-1, j].set_title("True high resolution")
        axs[-1, j].imshow(np.flipud(upsamp_hr[j].squeeze().to('cpu').numpy()))

    return mse, ssim, fig


def test_proc(model, ns, data_loader, model_name, up_shape=(64, 64), ensemble=1, architecture_details=None, denormalize=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_ssim = []
    all_mse = []
    all_sq_error = []
    all_bilinear_ssim = []
    all_bilinear_mse = []
    monthly_errors = defaultdict(list)

    for i, batch in enumerate(tqdm(data_loader)):
        high_res_imgs = batch["high_res"].unsqueeze(1).float()  # because one channel
        upsamp_hr = F.interpolate(high_res_imgs, size=up_shape, mode='bilinear').to(device)

        if denormalize:
            if architecture_details is None:
                raise ValueError("Architecture details must be provided to denormalize the data")
            upsamp_hr = denormalizing(upsamp_hr, float(architecture_details['Max_norm_high']), float(architecture_details['Min_norm_high']))
            data_range = float(architecture_details['Max_norm_high']) - float(architecture_details['Min_norm_high'])
            ssim_measure = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
        else:
            ssim_measure = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        low_res_imgs = batch["low_res"][:, 0, :, :].unsqueeze(1).float()
        upsampled = F.interpolate(low_res_imgs, size=up_shape, mode='bilinear').to(device)
        if denormalize:
            upsampled = denormalizing(upsampled, float(architecture_details['Max_norm_high']), float(architecture_details['Min_norm_high']))

        bilinear_mse = F.mse_loss(upsampled, upsamp_hr)
        bilinear_ssim = ssim_measure(upsampled, upsamp_hr)

        all_bilinear_ssim.append(bilinear_ssim.unsqueeze(0))
        all_bilinear_mse.append(bilinear_mse.unsqueeze(0))

        ensemble_all_x_t = [torch.stack(DDPM_infer(model, ns, batch, up_shape)) for _ in range(ensemble)]
        all_x_t = torch.stack(ensemble_all_x_t).mean(dim=0)
        x_t = all_x_t[-1]

        if denormalize:
            x_t = denormalizing(x_t, float(architecture_details['Max_norm_high']), float(architecture_details['Min_norm_high']))

        sq_error = torch.mean(((x_t - upsamp_hr) ** 2), dim=(0, 1))
        mse = F.mse_loss(x_t, upsamp_hr)
        ssim = ssim_measure(x_t, upsamp_hr)

        all_sq_error.append(sq_error)
        all_ssim.append(ssim.unsqueeze(0))
        all_mse.append(mse.unsqueeze(0))

        dates = batch["dates"]
        mse_per_image = torch.mean(((x_t - upsamp_hr) ** 2), dim=(1, 2, 3))
        for date, error in zip(dates, mse_per_image):
            dt = datetime.fromtimestamp(date.item())
            month_year = dt.strftime("%Y-%m")
            monthly_errors[month_year].append(error.item())

        if i == 0:
            plot_predictions(model_name, low_res_imgs, up_shape, upsamp_hr, x_t, ensemble_all_x_t)

    avg_bilinear_ssim = torch.mean(torch.cat(all_bilinear_ssim))
    avg_bilinear_mse = torch.mean(torch.cat(all_bilinear_mse))
    avg_sq_error = torch.mean(torch.stack(all_sq_error), dim=0)
    plot_error_image(avg_sq_error, model_name)
    avg_ssim = torch.mean(torch.cat(all_ssim))
    avg_mse = torch.mean(torch.cat(all_mse))
    plot_monthly_errors(monthly_errors, model_name)

    with open(f"test_results/{model_name}_results.txt", 'w') as file:
        file.write(f'MSE: {avg_mse}\n')
        file.write(f'SSIM: {avg_ssim}\n')
        file.write(f'Bilinear SSIM: {avg_bilinear_ssim}\n')
        file.write(f'Bilinear MSE: {avg_bilinear_mse}\n')


def plot_predictions(model_name, low_res_imgs, up_shape, upsamp_hr, x_t, ensemble_all_x_t):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_plots = 3
    fig, axs = plt.subplots(4, n_plots, figsize=(3 * n_plots, 12), gridspec_kw={'hspace': 0.3})

    for j in range(n_plots):
        axs[0, j].imshow(low_res_imgs[j].squeeze().to('cpu').numpy())
        axs[0, j].set_title("Low resolution image")

        axs[1, j].imshow(F.interpolate(low_res_imgs[j].unsqueeze(0), size=up_shape, mode='bilinear').squeeze().to('cpu').numpy())
        axs[1, j].set_title("Bilinear interpolation")

        axs[2, j].imshow(np.flipud(x_t[j].squeeze().to('cpu').numpy()))
        axs[2, j].set_title("Prediction")

        axs[3, j].imshow(np.flipud(upsamp_hr[j].squeeze().to('cpu').numpy()))
        axs[3, j].set_title("True high resolution")

    plt.savefig(f"test_results/{model_name}_predictions.png")
    plt.close()


def DDPM_infer(model, ns, batch, up_shape):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = batch["low_res"].shape[0]
    low_res_imgs = batch["low_res"].float()
    upsampled = F.interpolate(low_res_imgs, size=up_shape, mode='bilinear').to(device)

    model.to(device)
    n_steps = 4
    all_x_t = []

    x_t = torch.randn((batch_size, 1, up_shape[0], up_shape[1])).to(device)
    all_x_t.append(x_t)
    for i in range(ns.T - 1, -1, -1):
        t = torch.tensor([i for _ in range(batch_size)]).to(device).long()
        unet_input = torch.cat((x_t, upsampled), dim=1).float().to(device)

        betas_t = ns.get_index_from_list(ns.betas, t, x_t.shape).to(device)
        sqrt_one_minus_alphas_bar_t = ns.get_index_from_list(ns.sqrt_one_minus_alphas_bar, t, x_t.shape).to(device)
        sqrt_inv_alphas_t = ns.get_index_from_list(ns.sqrt_inv_alphas, t, x_t.shape).to(device)

        with torch.no_grad():
            pred_noise = model(unet_input, t).to(device)
            x_t_minus_1 = sqrt_inv_alphas_t * (x_t - betas_t * pred_noise / sqrt_one_minus_alphas_bar_t)
            noise_variance = ns.betas * (1.0 - F.pad(ns.alphas_bar[:-1], (1, 0), value=1.0)) / (1.0 - ns.alphas_bar)
            noise_variance_t = ns.get_index_from_list(noise_variance, t, x_t.shape).to(device)

            if i == 0:
                x_t = x_t_minus_1
            else:
                noise = torch.randn_like(x_t).to(device)
                x_t = x_t_minus_1 + torch.sqrt(noise_variance_t) * noise

            if i % (ns.T / n_steps) == 0:
                all_x_t.append(x_t)

    return all_x_t

def plot_error_image(error_tensor, model_name):
    error_np = error_tensor.to('cpu').numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(np.flipud(error_np), cmap='viridis')
    plt.colorbar(shrink=0.8)
    plt.savefig(f"test_results/{model_name}_error_image.png")
    plt.close()

def plot_monthly_errors(monthly_errors, model_name):
    months = sorted(monthly_errors.keys())
    values = [monthly_errors[month] for month in months]

    plt.figure(figsize=(10, 6))
    plt.boxplot(values)

    plt.xlabel('Year-Month')
    plt.ylabel('Error')
    plt.xticks(range(1, len(months) + 1), months, rotation=45)

    plt.tight_layout()
    plt.savefig(f"test_results/{model_name}_monthly_errors.png")
    plt.close()

