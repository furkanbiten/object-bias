from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
from lmdbdict import lmdbdict
from lmdbdict.methods import DUMPS_FUNC, LOADS_FUNC
import os
import numpy as np
import numpy.random as npr
import random
from functools import partial

import torch
import torch.utils.data as data

import multiprocessing
import six
import re
import pickle
import itertools
from ..utils.bias_utils import SentenceSimplifier, getClass2IdMap, getId2ClassMap, updateCovMatrix

class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.

    in_memory: if in_memory is True, we save all the features in memory
               For individual np(y|z)s, we don't need to do that because the system will do this for us.
               Should be useful for lmdb or h5.
               (Copied this idea from vilbert)
    """
    def __init__(self, db_path, ext, in_memory=False):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(six.BytesIO(x))
        else:
            def load_npz(x):
                x = np.load(six.BytesIO(x))
                return x['feat'] if 'feat' in x else x['z']  # normally it should be 'feat', but under cocotest_bu, the key is saved to be 'z' mistakenly.
            self.loader = load_npz
        if db_path.endswith('.lmdb'):
            self.db_type = 'lmdb'
            self.lmdb = lmdbdict(db_path, unsafe=True)
            self.lmdb._key_dumps = DUMPS_FUNC['ascii']
            self.lmdb._value_loads = LOADS_FUNC['identity']
        elif db_path.endswith('.pth'): # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        elif db_path.endswith('h5'):
            self.db_type = 'h5'
            self.loader = lambda x: np.array(x).astype('float32')
        else:
            self.db_type = 'dir'

        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}
    
    def get(self, key):

        if self.in_memory and key in self.features:
            # We save f_input because we want to save the
            # compressed bytes to save memory
            f_input = self.features[key]
        elif self.db_type == 'lmdb':
            f_input = self.lmdb[key]
        elif self.db_type == 'pth':
            f_input = self.feat_file[key]
        elif self.db_type == 'h5':
            f_input = h5py.File(self.db_path, 'r')[key]
        else:
            f_input = open(os.path.join(self.db_path, key + self.ext), 'rb').read()

        if self.in_memory and key not in self.features:
            self.features[key] = f_input

        # load image
        feat = self.loader(f_input)

        return feat

class Dataset(data.Dataset):
    
    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.seq_per_img = opt.seq_per_img
        
        # feature related options
        self.use_fc = getattr(opt, 'use_fc', True)
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)
        self.use_vc = getattr(opt, 'use_vc', 0)

        self.use_ft = getattr(opt, 'use_ft', 0)
        self.augmentation = getattr(opt, 'augmentation', 0)
        self.aligned = getattr(opt, 'label_aligned', 0)
        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        if 'ix_to_word' in self.info:
            self.ix_to_word = self.info['ix_to_word']
            self.word_to_ix = {v: k for k, v in self.ix_to_word.items()}
            self.vocab_size = len(self.ix_to_word)
            print('vocab size is ', self.vocab_size)
        
        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir, opt.input_label_h5)
        """
        Setting input_label_h5 to none is used when only doing generation.
        For example, when you need to test on coco test set.
        """
        if self.opt.input_label_h5 != 'none':
            self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
            # load in the sequence data
            seq_size = self.h5_label_file['labels'].shape
            self.label = self.h5_label_file['labels'][:]
            self.seq_length = seq_size[1]
            print('max sequence length in data is', self.seq_length)
            # load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.h5_label_file['label_start_ix'][:]
            self.label_end_ix = self.h5_label_file['label_end_ix'][:]
        else:
            self.seq_length = 1

        self.data_in_memory = getattr(opt, 'data_in_memory', False)
        self.fc_loader = HybridLoader(self.opt.input_fc_dir, '.npy', in_memory=self.data_in_memory)
        self.att_loader = HybridLoader(self.opt.input_att_dir, '.npz', in_memory=self.data_in_memory)
        self.box_loader = HybridLoader(self.opt.input_box_dir, '.npy', in_memory=self.data_in_memory)
        if self.use_vc:
            self.att_loader_vc = HybridLoader(self.opt.input_att_dir_vc, '.npy')

        if self.use_ft:
            self.att_loader_ft = HybridLoader(self.opt.input_att_dir_ft, '.npz', in_memory=self.data_in_memory)
            if self.augmentation:
                self.simplifier = SentenceSimplifier()
                self.obj_ids_loader = HybridLoader(self.opt.obj_ids_dir, '.npy', in_memory=self.data_in_memory)
                self.class2id = pickle.load(open('data/object_names', 'rb'))
                # THIS FUCKING SHIT IS UGLY I KNOW, BUT FUCK MSCOCO SERIOUSLY!!!!
                self.class2idMap = getClass2IdMap()
                self.id2classMap = getId2ClassMap()
                self.id2class = {v:k for k,v in self.class2id.items()}
                self.covMatrix = pickle.load(open('data/covMatrix/covMatrixAnns.pkl', 'rb'))
                self.ft_repr = pickle.load(open('data/objects_word_repr', 'rb'))

        self.num_images = len(self.info['images']) # self.label_start_ix.shape[0]
        print('read %d image features' %(self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if not 'split' in img:
                self.split_ix['train'].append(ix)
                self.split_ix['val'].append(ix)
                self.split_ix['test'].append(ix)
            elif img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))

    def inv_multinomial(self, pair):
        cls_idx = self.class2idMap[pair[0]]
        class_probabilities = 1 / (self.covMatrix[cls_idx, :] / np.sum(self.covMatrix[cls_idx, :]))
        class_probabilities[cls_idx] = 0
        class_probabilities = class_probabilities / np.sum(class_probabilities)
        replacement_word = self.id2classMap[np.argmax(np.random.multinomial(1, class_probabilities))]
        return replacement_word

    def replace_words(self, synonym_pair, replacement_word, simplified_string):
        simplified_string = re.sub(r"\b{}\b".format(synonym_pair[1]), " {} ".format(replacement_word),
                                   simplified_string)
        simplified_string = re.sub(r" nt", "nt", simplified_string)
        simplified_string = simplified_string.strip()
        cap_ids = [int(self.word_to_ix[i]) for i in simplified_string.split()]
        return cap_ids

    def augment(self, img_id, tmp_seq):
        # try:
        obj_ids = self.obj_ids_loader.get(str(img_id))
        # except:
        #     obj_ids = []
        # For the moment it works only with 1 caption per image, i.e. seq_per_img must be 1
        caption_string = ' '.join([self.ix_to_word[str(i)] for i in tmp_seq[0] if i != 0])
        obj_names = [self.id2class[i] for i in obj_ids]
        class_combinations = list(set(itertools.combinations(obj_names, 2)))
        success, pair, synonym_pair, simplified_string = self.simplifier.simplify(caption_string, class_combinations)
        # if success:
        # if success and np.random.rand() < 0.5:
        # if success == True and np.random.rand() < 0.5 and pair[0] in self.class2idMap.keys() and pair[1] in self.class2idMap.keys():
        if np.random.rand() < 0.5 and pair[0] in self.class2idMap.keys() and pair[1] in self.class2idMap.keys():
            wordToReplace = pair[1]
            replacement_word = wordToReplace
            while replacement_word == pair[0] or replacement_word == wordToReplace:
                if self.augmentation == 'uniform':
                    replacement_word = np.random.choice(list(self.id2class.values()), 1)[0]
                elif self.augmentation == 'multinomial':
                    replacement_word = self.inv_multinomial(pair)
                elif self.augmentation == 'co-occurence':
                    replacement_word = self.inv_multinomial(pair)
                    cls_pair = (self.class2idMap[wordToReplace], self.class2idMap[replacement_word])
                    self.covMatrix = updateCovMatrix(self.covMatrix, set(obj_names), cls_pair, self.class2idMap)

            # Using simplified caption
            cap_ids = self.replace_words(synonym_pair, replacement_word, simplified_string)
            # Using original caption
            # cap_ids = self.replace_words(synonym_pair, replacement_word, caption_string)

            return cap_ids, obj_ids, wordToReplace, replacement_word, pair
        else:
            return None, None, None, None

    def convert_str(self, ids):
        return ' '.join([self.ix_to_word[str(i)] for i in ids if i != 0])

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.label[ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def collate_func(self, batch, split):
        seq_per_img = self.seq_per_img

        fc_batch = []
        att_batch = []
        label_batch = []
        if self.use_ft and not self.aligned:
            ft_batch = []

        wrapped = False

        infos = []
        gts = []

        for sample in batch:
            # fetch image
            tmp_fc, tmp_att, tmp_seq, \
                ix, it_pos_now, tmp_wrapped = sample

            img_id = self.info['images'][ix]['id']
            new_tmp_seq = None
            if self.use_ft and not self.aligned:
                tmp_att, tmp_ft = tmp_att

                if self.augmentation and split == 'train':
                    try:
                        new_tmp_seq, obj_ids, word_to_replace, replacement_word, pair = self.augment(img_id, tmp_seq)
                    except Exception as e:
                        # print(e)
                        new_tmp_seq = None

                    if new_tmp_seq:
                        tmp_seq = np.zeros([self.seq_per_img, self.seq_length], dtype='int')
                        for q in range(self.seq_per_img):
                            tmp_seq[q, :min(len(new_tmp_seq), self.seq_length)] = new_tmp_seq[:self.seq_length]

                        indexes = [index for index, id in enumerate(obj_ids) if id == self.class2id[word_to_replace]]
                        for index in indexes:
                            tmp_ft[index] = self.ft_repr[self.class2id[replacement_word]]

                ft_batch.append(tmp_ft)

            tmp_label = np.zeros([seq_per_img, self.seq_length + 2], dtype='int')
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                tmp_label[:, 1 : self.seq_length + 1] = tmp_seq
            label_batch.append(tmp_label)

            if tmp_wrapped:
                wrapped = True

            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)

            # Used for reward evaluation
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                if self.augmentation and new_tmp_seq and split == 'train':
                    # word_to_replace_ix = int(self.word_to_ix[word_to_replace])
                    # replacement_word_ix = int(self.word_to_ix[replacement_word])
                    num_cap = self.label_end_ix[ix] - (self.label_start_ix[ix] - 1 )
                    new_gt = np.zeros((num_cap, self.seq_length))
                    for i, c in enumerate(self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]]):
                        success, _, synonym_pair, simplified_string = self.simplifier.simplify(self.convert_str(c), [pair])
                        if success:
                            # new_gt.append(self.replace_words(synonym_pair, replacement_word, simplified_string))
                            simple = self.replace_words(synonym_pair, replacement_word, simplified_string)
                            new_gt[i, :len(simple)] = np.array(simple)[:self.seq_length]
                    gts.append(new_gt)
                else:
                    gts.append(self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
            else:
                gts.append([])
        
            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = img_id
            info_dict['file_path'] = self.info['images'][ix].get('file_path', '')
            infos.append(info_dict)

        # #sort by att_feat length
        # THIS IS WRONG AND COMPLETELY UNNECESSARY! SIIIIGH!!!!!
        # fc_batch, att_batch, label_batch, gts, infos = \
        #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
        # if self.use_ft:
        #     fc_batch, att_batch, ft_batch, label_batch, gts, infos = \
        #         zip(*sorted(zip(fc_batch, att_batch, ft_batch, label_batch, gts, infos), key=lambda x: 0, reverse=True))
        # else:
        #     fc_batch, att_batch, label_batch, gts, infos = \
        #         zip(*sorted(zip(fc_batch, att_batch, label_batch, gts, infos), key=lambda x: 0, reverse=True))
        data = {}
        if self.use_ft and not self.aligned:
            max_att_len = max([_.shape[0] for _ in ft_batch])
            data['ft_feats'] = np.zeros([len(ft_batch), max_att_len, ft_batch[0].shape[1]], dtype='float32')
            for i in range(len(ft_batch)):
                data['ft_feats'][i, :ft_batch[i].shape[0]] = ft_batch[i]
            data['ft_masks'] = np.zeros(data['ft_feats'].shape[:2], dtype='float32')
            for i in range(len(ft_batch)):
                data['ft_masks'][i, :ft_batch[i].shape[0]] = 1
            # set att_masks to None if attention features have same length
            if data['ft_masks'].sum() == data['ft_masks'].size:
                data['ft_masks'] = None

        data['fc_feats'] = np.stack(fc_batch)
        # merge att_feats
        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch), max_att_len, att_batch[0].shape[1]], dtype='float32')
        for i in range(len(att_batch)):
            data['att_feats'][i, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        data['labels'] = np.vstack(label_batch)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['labels'])))
        mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype='float32')
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch
        data['labels'] = data['labels'].reshape(len(batch), seq_per_img, -1)
        data['masks'] = data['masks'].reshape(len(batch), seq_per_img, -1)

        data['gts'] = gts # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': it_pos_now, # the it_pos_now of the last sample
                          'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor

        return data

    def get_bbox(self, ix):
        box_feat = self.box_loader.get(str(self.info['images'][ix]['id']))
        # devided by image width and height
        x1, y1, x2, y2 = np.hsplit(box_feat, 4)
        h, w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
        box_feat = np.hstack((x1 / w, y1 / h, x2 / w, y2 / h, (x2 - x1) * (y2 - y1) / (w * h)))  # question? x2-x1+1??
        if self.norm_box_feat:
            box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
        return box_feat

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix, it_pos_now, wrapped = index #self.split_ix[index]
        if self.use_att:
            try:
                att_feat = self.att_loader.get(str(self.info['images'][ix]['id']))
                if self.use_vc:
                    att_feat_vc = self.att_loader_vc.get(str(self.info['images'][ix]['id']))
                    assert att_feat.shape[0] == att_feat_vc.shape[0]
                    att_feat = np.hstack((att_feat, att_feat_vc))
            except:
                att_feat = np.zeros((1, self.opt.att_feat_size), dtype=np.float32)

            if self.use_ft:
                # This try/except is put because some ground truth items in COCO is empty
                try:
                    att_feat_ft = self.att_loader_ft.get(str(self.info['images'][ix]['id']))
                    if att_feat_ft.shape[0]==0: att_feat_ft = np.zeros((1, 300), dtype=np.float32)
                except:
                    att_feat_ft = np.zeros((1, 300), dtype=np.float32)

                if self.use_box:
                    # Exactly the same reason as above
                    try:
                        box_feat = self.get_bbox(ix)
                        att_feat_ft = np.hstack([att_feat_ft, box_feat])
                    except:
                        att_feat_ft = np.hstack([att_feat_ft, np.zeros((att_feat_ft.shape[0], 5))])

                if self.aligned:
                    if att_feat.shape[0] != att_feat_ft.shape[0]:
                        att_feat = np.concatenate((att_feat, np.zeros((att_feat.shape[0], 300), dtype=np.float32)), axis=-1)
                    else:
                        att_feat = np.concatenate((att_feat, att_feat_ft), axis=-1)
                else:
                    att_feat = (att_feat, att_feat_ft)
            else:
                # Reshape to K x C
                att_feat = att_feat.reshape(-1, att_feat.shape[-1])

            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)

            if self.use_box and not self.use_ft:
                # box_feat = self.box_loader.get(str(self.info['images'][ix]['id']))
                # # devided by image width and height
                # x1,y1,x2,y2 = np.hsplit(box_feat, 4)
                # h, w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
                # box_feat = np.hstack((x1/w, y1/h, x2/w, y2/h, (x2-x1)*(y2-y1)/(w*h))) # question? x2-x1+1??
                # if self.norm_box_feat:
                #     box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
                box_feat = self.get_bbox(ix)
                att_feat = np.hstack([att_feat, box_feat])
                # sort the features by the size of boxes
                att_feat = np.stack(sorted(att_feat, key=lambda x:x[-1], reverse=True))

        else:
            att_feat = np.zeros((0,0), dtype='float32')
        if self.use_fc:
            try:
                fc_feat = self.fc_loader.get(str(self.info['images'][ix]['id']))
            except:
                # Use average of attention when there is no fc provided (For bottomup feature)
                if self.use_ft and not self.aligned:
                    fc_feat = att_feat[0].mean(0)
                else:
                    fc_feat = att_feat.mean(0)
        else:
            fc_feat = np.zeros((0), dtype='float32')
        if hasattr(self, 'h5_label_file'):
            seq = self.get_captions(ix, self.seq_per_img)
        else:
            seq = None
        return (fc_feat,
                att_feat, seq,
                ix, it_pos_now, wrapped)

    def __len__(self):
        return len(self.info['images'])

class DataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.dataset = Dataset(opt)

        # Initialize loaders and iters
        self.loaders, self.iters = {}, {}
        for split in ['train', 'val', 'test']:
            if split == 'train':
                sampler = MySampler(self.dataset.split_ix[split], shuffle=True, wrap=True)
            else:
                sampler = MySampler(self.dataset.split_ix[split], shuffle=False, wrap=False)
            self.loaders[split] = data.DataLoader(dataset=self.dataset,
                                                  batch_size=self.batch_size,
                                                  sampler=sampler,
                                                  pin_memory=True,
                                                  num_workers=getattr(opt, 'num_workers', 4), # 4 is usually enough
                                                  collate_fn=partial(self.dataset.collate_func, split=split),
                                                  drop_last=False)
            self.iters[split] = iter(self.loaders[split])

    def get_batch(self, split):
        try:
            data = next(self.iters[split])
        except StopIteration:
            self.iters[split] = iter(self.loaders[split])
            data = next(self.iters[split])
        return data

    def reset_iterator(self, split):
        self.loaders[split].sampler._reset_iter()
        self.iters[split] = iter(self.loaders[split])

    def get_vocab_size(self):
        return self.dataset.get_vocab_size()

    @property
    def vocab_size(self):
        return self.get_vocab_size()

    def get_vocab(self):
        return self.dataset.get_vocab()

    def get_seq_length(self):
        return self.dataset.get_seq_length()

    @property
    def seq_length(self):
        return self.get_seq_length()

    def state_dict(self):
        def get_prefetch_num(split):
            if self.loaders[split].num_workers > 0:
                return (self.iters[split]._send_idx - self.iters[split]._rcvd_idx) * self.batch_size
            else:
                return 0
        return {split: loader.sampler.state_dict(get_prefetch_num(split)) \
                    for split, loader in self.loaders.items()}

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        for split in self.loaders.keys():
            self.loaders[split].sampler.load_state_dict(state_dict[split])


class MySampler(data.sampler.Sampler):
    def __init__(self, index_list, shuffle, wrap):
        self.index_list = index_list
        self.shuffle = shuffle
        self.wrap = wrap
        # if wrap, there will be not stop iteration called
        # wrap True used during training, and wrap False used during test.
        self._reset_iter()

    def __iter__(self):
        return self

    def __next__(self):
        wrapped = False
        if self.iter_counter == len(self._index_list):
            self._reset_iter()
            if self.wrap:
                wrapped = True
            else:
                raise StopIteration()
        if len(self._index_list) == 0: # overflow when 0 samples
            return None
        elem = (self._index_list[self.iter_counter], self.iter_counter+1, wrapped)
        self.iter_counter += 1
        return elem

    def next(self):
        return self.__next__()

    def _reset_iter(self):
        if self.shuffle:
            rand_perm = npr.permutation(len(self.index_list))
            self._index_list = [self.index_list[_] for _ in rand_perm]
        else:
            self._index_list = self.index_list

        self.iter_counter = 0

    def __len__(self):
        return len(self.index_list)

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        self._index_list = state_dict['index_list']
        self.iter_counter = state_dict['iter_counter']

    def state_dict(self, prefetched_num=None):
        prefetched_num = prefetched_num or 0
        return {
            'index_list': self._index_list,
            'iter_counter': self.iter_counter - prefetched_num
        }

    