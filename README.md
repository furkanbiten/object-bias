# Reducing Object Hallucination in Image Captioning
Welcome! 

In this repo, you can find the code for our paper. 
![Object-Bias Model!](https://github.com/furkanbiten/object-bias/blob/master/models.pdf)

## Creating the environment
First and foremost, let's start by creating the environment so that 
we are on the same page with all the different library versions so that you hate me less.
Here is the command:
`conda env create -f req.yml --name object-bias`

This might take a while and patience is a virtue.

Then we activate the environment.

`conda activate object-bias`

## Data Setup
I tried my best to ease the most boring process, but one can only do so much.

All the data can be found [here](https://cvcuab-my.sharepoint.com/personal/abiten_cvc_uab_cat/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fabiten%5Fcvc%5Fuab%5Fcat%2FDocuments%2Fobject%2Dbias%2Fdata)
Download ALL the data from there and put it in the `data/` folder and unzip it. 

Some files are taken from ruotianluo's repo called [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch),
so just in case if something goes wrong, you can find some of these files in [here](https://github.com/ruotianluo/self-critical.pytorch/blob/master/data/README.md).

## Where is where?
This section is to get you up to speed about the code and 
which part does what (I admit that the title name is a bit unorthodox). 
In `tools` folder, you have the train and eval code. 

In `captioning/models`, you can find the model changes we did.

In `captioning/data/dataloader.py`, in line 203, you can find how we augment the sentences in dataloader.

In `captioning/utils/bias_utils.py`, you can find the code for sentence simplification and 
how we implement the co-occurence updating with sentence simplification. 
In theory, `SentenceSimplifier` class should be standalone code, meaning it can be plugged into another code seamlessly (although, nothing is that easy!).


## Running the code
Before we talk about running the code, couple of important points on the augmentation. 
We train a model with object label concatenated. 
To do the training with augmentation, we always start from this model weight 
(the names are updown_ft_concat, aoa_ft_concat, see the [Model Weights](##Model Weights)).

So, finally, the exciting staff. To run the code for training, you need to run the command in terminal. Let me first give an example.

`python tools/train.py --cfg configs/updown/updown_ft_concat_aug.yml --augmentation co-occurence --checkpoint_path models/updown_ft_concat_occur --start_from models/updown_ft_concat/ --max_epochs 60 --save_checkpoint_every 10000`

Let's break it down. `--cfg` is the configuration file we want to use, 
`--augmentation` is the type of augmentation you want to do for sentence simplification, 
`--checkpoint_path` where to save, `--start_from` which model to load. 

## Model Weights
All the model configuration is already in this repo in `configs/`. 
All the model weights can be downloaded [here](https://cvcuab-my.sharepoint.com/personal/abiten_cvc_uab_cat/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fabiten%5Fcvc%5Fuab%5Fcat%2FDocuments%2Fobject%2Dbias%2Fmodels)

## Evaluation 
### On Classic Captioning Metrics
`python tools/eval.py --dump_images 0 --num_images 5000 --model models/updown_ft_concat_occur_noise/model.pth --infos_path models/updown_ft_concat_occur_noise/infos_.pkl --language_eval 1 --input_att_dir_ft data/ft_frcnn_att`

### On CHAIR metric

## Conclusion
As usual, sorry for the bugs and unnecessary part of the code,
I blame the conference deadlines!

To err is human.