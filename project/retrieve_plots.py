import wandb
import pandas as pd
import matplotlib.pyplot as plt

# Configure global settings for matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 20
})

# Initialize Weights & Biases API
api = wandb.Api()


measure_names = {
    "loss": "Loss",
    "DDPM_ssim": "SSIM",
    "DDIM_ssim": "SSIM",
    "DDPM_error": "MSE",
    "DDIM_error": "MSE",
    "time": "Time"
}

def fetch_run_ids(run_names, entity="WindDownscaling", project="Report"):
    """ Fetch IDs for specified run names from Weights & Biases. """
    runs = api.runs(f"{entity}/{project}")
    return [run.id for run in runs if run.name in run_names]

def plot_results(run_names, labels, eval_set="train", eval_measure="loss", add_info="", entity="WindDownscaling", project="Report"):
    """ Generate and save plots for specified runs and measures. """
    metric_to_evaluate = f"{eval_set}/{eval_measure}"
    plt.figure(figsize=(10, 6))
    min_epochs = float('inf')
    run_ids = fetch_run_ids(run_names, entity, project)

    for run_id, label in zip(run_ids, labels):
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history(keys=['epoch', metric_to_evaluate])
        history_df = pd.DataFrame(history)
        max_epochs = max(history_df['epoch'])
        min_epochs = min(min_epochs, max_epochs)
        plt.plot(history_df['epoch'], history_df[metric_to_evaluate], label=label)

    plt.xlabel('Epoch')
    plt.xlim(0, min_epochs)
    plt.ylabel(measure_names.get(eval_measure, "Value"))
    plt.legend()
    plt.grid(True)

    plot_name = f"{eval_set}_{eval_measure}"
    if add_info:
        plot_name += f"_{add_info}"
    plt.savefig(f"train_plots/{plot_name}.pdf")
    plt.close()

if __name__ == "__main__":
    entity, project = "WindDownscaling", "Report"
    run_names = ["T200_PS3_lr0.0001_B64", "T50_PS3_lr0.0001_B64", "T25_PS3_lr0.0001_B64"]
    labels = ["S = 200", "S = 50", "S = 25"]
    eval_sets = ["train", "val"]
    eval_measures = ["time", "DDPM_ssim"]

    for eval_set in eval_sets:
        for eval_measure in eval_measures:
            plot_results(run_names,labels, eval_set=eval_set, eval_measure=eval_measure,add_info="", entity=entity, project=project)


