import os
import json
import warnings
import argparse
from torch.utils.data import DataLoader
from models import DenoisingUnet
from evaluate import eval_proc, test_proc
from train import train_proc
from utils import NoiseScheduler, load_data, load_model

# Ignore warnings
warnings.filterwarnings("ignore")

# Wandb config
os.environ["WANDB_ENTITY"] = "WindDownscaling"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--auto_name", default=False, action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--setup_path", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-04)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--pred_step", type=int, default=0)
    parser.add_argument("--attention", default=False, action="store_true")
    parser.add_argument("--big", type=bool, default=False)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--gpu", nargs='+', type=int)
    args = parser.parse_args()

    if args.auto_name:
        args.run_name = f"T{args.T}_PS{args.pred_step}_lr{args.lr}_B{args.batch_size}_E{args.ensemble}"
        if args.dropout:
            args.run_name += f"_dr{args.dropout}"
        if args.big:
            args.run_name += "_big"
        if args.attention:
            args.run_name += "_A"

    print("Loading of the data")
    train_dataset, in_norm, out_norm = load_data(['u10', 'v10'], 2010, 2018, args.pred_step, normalize=True)
    print("Max and min values of the data: ", in_norm, out_norm)
    
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset, _, _ = load_data(
        ['u10', 'v10'],
        2019, 
        2019, 
        args.pred_step, 
        normalize=True, 
        in_max=in_norm[0], 
        in_min=in_norm[1], 
        out_max=out_norm[0], 
        out_min=out_norm[1],
    )
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset, _, _ = load_data(
        ['u10', 'v10'],
        2020, 
        2020, 
        args.pred_step, 
        normalize=True, 
        in_max=in_norm[0], 
        in_min=in_norm[1], 
        out_max=out_norm[0], 
        out_min=out_norm[1],
    )
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    print("Data loaded")
    
    # Define the model
    print("Model creation")
    unet_channels = (64, 128, 256, 512, 1024) if args.big else (64, 128, 256, 512)
    kernel_sizes = [3, 3, 3, 3]
    input_channels = 2 + args.pred_step
    output_channels = 1
    time_emb_dim = 32
    up_shape = (64, 64)

    model = DenoisingUnet(
        unet_channels,
        input_channels,
        output_channels,
        time_emb_dim,
        attention=args.attention,
        dropout=args.dropout,
    )

    min_noise = 0.015
    max_noise = 0.95
    ns = NoiseScheduler(args.T, min_noise, max_noise)
    
    print("Model created")
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    architecture_details = {
        "Max_norm_high": str(in_norm[0]),
        "Min_norm_high": str(in_norm[1]),
        "Max_norm_low": str(out_norm[0]),
        "Min_norm_low": str(out_norm[1]),
        "Min noise": min_noise,
        "Max noise": max_noise,
        "T": args.T,
        "Epochs": args.num_epochs,
        "Batch size": args.batch_size,
        "lr": args.lr,
        "ensemble": args.ensemble,
        "dropout": args.dropout,
        "unet_channels": unet_channels,
        "kernel_sizes": kernel_sizes,
        "input_channels": input_channels,
        "time_emb_dim": time_emb_dim, 
        "Attention" : args.attention
    }

    if args.train:
        print("Start of the training process")
        train_proc(
            model, 
            ns, 
            train_data_loader, 
            val_data_loader,
            architecture_details,
            up_shape=up_shape,
            num_epochs=args.num_epochs,
            lr=args.lr,
            ensemble=args.ensemble,
            run_name=args.run_name,
            save=True,
            load_best_model=True
        )
        print("End of the training process")

    if args.test:
        print("Testing of the saved model")
        model = load_model(model, args.model_path)
        architecture_data_path = args.model_path.replace(".pth", "_architecture_data.json")
        with open(architecture_data_path, 'r') as fp:
            architecture_details = json.load(fp)
        model.eval()
        print("Testing using DDPM...")
        test_proc(
            model, 
            ns, 
            test_data_loader, 
            model_name=args.run_name, 
            up_shape=up_shape, 
            ensemble=args.ensemble, 
            architecture_details=architecture_details, 
            denormalize=True
        )
        print("End of the testing process")

if __name__ == "__main__":
    main()