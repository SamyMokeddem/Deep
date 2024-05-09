import os
from torch.utils.data import  DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import DenoisingUnet, SimpleUnet
from evaluate import test_proc
from train import train_proc
from utils import NoiseScheduler, load_data, load_model
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Wandb config
import wandb
os.environ["WANDB_ENTITY"]="WindDownscaling"


if __name__ == "__main__":
    train = True
    test = False

    num_epochs = 500

    print("loading of the data")
    train_dataset, test_dataset, val_dataset = load_data(['u10', 'v10'], 2010, 2019, split_ratio = [0.1, 0.1])

    batch_size = 128

    train_data_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle=True)

    print("Data loaded")
    # Define the model
    print("Model creation")
    unet_channels = (64, 128, 256, 512) # (64, 128, 256, 512, 1024) # <-- From youtube video. In the paper: [64, 128, 256, 384]
    kernel_sizes = [3, 3, 3, 3]
    input_channels = 2
    output_channels = 1
    time_emb_dim = 32 # I guess that's a standard value
    dropout = 0.2
    lr = 1e-04
    up_shape = (64, 64)
    
    attention = "Spatial only"

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
            time_emb_dim,
            dropout=dropout
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
        "time_emb_dim": time_emb_dim, 
        "Attention" : attention
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