# Reducing Object Hallucination in Image Captioning
Welcome! In this repo, you can find the code for [our paper](https://arxiv.org/abs/2110.01705). 
![Object-Bias Model!](https://github.com/furkanbiten/object-bias/blob/master/models.jpg)

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

All the data can be found [here](https://cvcuab-my.sharepoint.com/:f:/g/personal/abiten_cvc_uab_cat/ErGQh6BUjORCkTBw8R6VEVQBk-TNYJxbSGwWjYRhPgXTCQ?e=qglpbQ). 
Download ALL the data from there and put it in the `data/` folder and unzip it. 

`cocobu_att.zip`: FRCNN extracted features per image.<br/>
`fasttext_att.zip`: Fasttext representation of the GROUND TRUTH object labels per image.<br/>
`obj_ids.zip`: Mapping of each object ids of COCO per image.<br/>
`ft_frcnn_att`: This folder is inside the data.zip, it is the FastText representation of FRCNN extracted object labels.<br/>

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
Before we talk about running the code, couple of important points on the training procedure. 
We first train a model with the concatenation of object labels. 
To do the training with augmentation, we always start from this model weight 
(the names are updown_ft_concat, aoa_ft_concat, see the [Model Weights](#Model Weights)).

So, finally, the exciting staff. To run the code for training, you need to run the command in terminal. Let me first give an example.

`python tools/train.py --cfg configs/updown/updown_ft_concat_aug.yml --augmentation co-occurence --checkpoint_path models/updown_ft_concat_occur --start_from models/updown_ft_concat/ --max_epochs 60 --save_checkpoint_every 10000`

Let's break it down. <br/>
`--cfg` is the configuration file we want to use. <br/>
`--augmentation` is the type of augmentation we want to do for object bias.<br/> 
`--checkpoint_path` where to save.<br/>
`--start_from` which model to load. <br/>

## Model Weights
All the model configuration is already in this repo in `configs/`. 
All the model weights can be downloaded [here](https://cvcuab-my.sharepoint.com/:f:/g/personal/abiten_cvc_uab_cat/ErrTNrBc9ydLkkFfiJQrf5IB7Gt2tSL4d9zjHpZEh3uavQ?e=l3uuaW)

## Evaluation 
Evaluation is seperated into 2 blocks. First is the classic image captioning metrics; SPICE, CIDER, BLEU, etc.
And then, we use and integrate the publicly available CHAIR [code](https://github.com/LisaAnne/Hallucination).

### On Classic Captioning Metrics
Once you train your model, you can simply evaluate the model by running:

`python tools/eval.py --dump_images 0 --num_images 5000 --model [CHOOSE model_best.pth] --infos_path [CHOOSE infos_best] --language_eval 1 --input_att_dir_ft data/ft_frcnn_att`

All the evaluation is done Karpathy split. Special mention for `--input_att_dir_ft data/ft_frcnn_att`. 
If you do not specifically give this argument, the model will be evaluated with ground truth object labels. 
The one that is specified is obtained from FRCNN.


### On CHAIR metric
After you run the previous command `tools/eval.py`, this will produce you a json file in the folder `eval_results`.
Now, we will use that json file to calculate the CHAIRi and CHAIRs scores. 

`python chair.py --cap_file [JSON FILE IN THE EVAL_RESULTS]`

For the more interested parties (I am dreaming, I know!), 
you can find all the scores/sentences of every model for the test set in [here.](https://cvcuab-my.sharepoint.com/:f:/g/personal/abiten_cvc_uab_cat/Ev1yZMJVBSJOmosy8lJ7fSgB6boboxHE-zW9I2FglPaKGg?e=kPI0jG) 
## Conclusion
As usual, sorry for the bugs and unnecessary part of the code,
I blame the conference deadlines!

To err is human.