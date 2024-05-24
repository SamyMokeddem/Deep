Sure, here is a template for a README file that includes command parameter usage:

---

# Project Name

Brief description of what the project does.

## Installation

Instructions on how to install the project.

```bash
# Example for a Python project
git clone https://github.com/username/project-name.git
cd project-name
pip install -r requirements.txt
```

## Usage

General instructions on how to use the project.

```bash
# Example usage
python weather_diffusion.py [options]
```

## Command Parameters

Detailed explanation of the command parameters.

```
--train             When call this parameter will activate the training process             
--test              when call this parameter will activate the test process   
--auto_name         when call this parameter will set a auto_name to the run according to the hyperparameters
--run_name=[str]    Give the name of the string to the run_name
--model_path=[str]  Give the path of the model to load for the test process     
--setup_path=[str]  Give the path of the model parameters for the test process
--lr=[float]        Give the learning used during the training
--ensemble=[int]    Give the number of inference used for the ensemble method
--dropout=[float]   give the dropout used during the training
--batch_size=[int]  give the batch_size used during the training and the testing
--pred_step=[int]   give the number of previous low res image used as input
--attention         enable the attention module in the model
--big               add a supplementary stage to the unet
--T=[int]           give the number of noise step used to train the model and infer from the model
--gpu=[int]         give the id of the gpu to use for the training

```
## Examples

Examples of how to run the project with different parameters.

```bash
# training
python weather_diffusion.py --train --batch_size=64 --pred_step=3 --attention --ensemble=2 --T=50
# test
python weather_diffusion.py --test --batch_size=64 --pred_step="train_models/T50_PS3_lr0.0001_B128_E3_best.pth" --ensemble=3 --pred_step=3 --T=50
```
