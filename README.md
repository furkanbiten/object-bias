# Reducing Object Hallucination in Image Captioning
Welcome! 

In this repo, you can find the code for our paper. 
![Object-Bias Model!](https://github.com/furkanbiten/object-bias/blob/master/models.pdf)

## Creating the environment
First and foremost, let's start by creating the environment so that 
we are on the same page with all the different library versions and you hate me less.
Here is the command:
`conda create --name object-bias --file requirements.txt`

This might take a while and patience is a virtue.

## Data Setup
Now, this section is a bit tricky. You will need to download quite a lot of files. 
I tried my best to ease the process, but one can only do so much.

Since this repo is building on top of ruotianluo's repo called [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch), 
most of the data can be found in there. 

All the files will follow the same scheme, download, extract and put it in the `data` folder, this process is also known as DEP.
Annotation files can be found in [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) and DEP it.

Since current methods use Faster-RCNN extracted features, 
those are graciously provided by ruotianluo in [here](https://drive.google.com/file/d/1hun0tsel34aXO4CYyTRIvHJkcbZHwjrD/view), then DEP it.

Also, DEP the files named `cocotalk_label.h5`, `cocotalk.json` and `coco-train-idxs.p` from [here](https://drive.google.com/drive/folders/1eCdz62FAVCGogOuNhy87Nmlo5_I0sH2J), 
again provided by ruotianluo.

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
`python tools/train.py --cfg configs/updown/updown_ft_concat_aug.yml --augmentation co-occurence --checkpoint_path models/updown_ft_concat_occur_noise --start_from models/updown_ft_concat/ --max_epochs 60 --save_checkpoint_every 10000`
`python tools/eval.py --dump_images 0 --num_images 5000 --model models/updown_ft_concat_occur_noise/model.pth --infos_path models/updown_ft_concat_occur_noise/infos_.pkl --language_eval 1 --input_att_dir_ft data/ft_frcnn_att`

## Conclusion
As usual, sorry for the bugs and unnecessary part of the code,
I blame the conference deadlines!

To err is human.