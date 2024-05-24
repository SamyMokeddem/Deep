import wandb
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 20
})

api = wandb.Api()

measure_names = {}
measure_names["loss"] = "Loss"
measure_names["DDPM_ssim"] = "SSIM"
measure_names["DDIM_ssim"] = "SSIM"
measure_names["DDPM_error"] = "MSE"
measure_names["DDIM_ssim"] = "MSE"
measure_names["time"] = "Time"

def names_to_ids(run_names, entity="WindDownscaling", project="Report"):
    runs = api.runs(entity + "/" + project)
    
    run_ids = [] 
    for run in runs:
        if run.name in run_names:
            run_ids.append(run.id)
            
    return run_ids

def plot_results(run_names, labels, eval_set="train", eval_measure="loss", add_info="", entity="WindDownscaling", project="Report"):
    to_eval = eval_set + "/" + eval_measure
    
    plt.figure(figsize=(10, 6))
    
    min_epochs = float("inf")
    
    run_ids = names_to_ids(run_names)

    for run_id, label in zip(run_ids, labels):  
        run = api.run(f"{entity}/{project}/{run_id}")
        
        history = run.history(keys=['epoch', to_eval])
        
        history_df = pd.DataFrame(history)
        
        max_epochs = max(history_df['epoch'])
        if max_epochs < min_epochs:
            min_epochs = max_epochs

        plt.plot(history_df['epoch'], history_df[to_eval], label=label)  

    plt.xlabel('Epoch')
    plt.xlim(0, min_epochs)
    plt.ylabel(measure_names[eval_measure])
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
    
    labels = ["S = 25", "S = 50", "S = 200"] 
    
    eval_sets = ["train",
                 "val"]

    eval_measures = ["time", "DDPM_ssim"]
    
    for eval_set in eval_sets:
        for eval_measure in eval_measures:
            plot_results(run_names,labels, eval_set=eval_set, eval_measure=eval_measure, entity=entity, project=project)