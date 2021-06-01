import argparse
import json
import os
from pycocotools.coco import COCO
import fasttext
import pickle
import numpy as np
import tqdm


def create_folder(root, path):
    if not os.path.exists(os.path.join(root, path)):
        os.makedirs(os.path.join(root, path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--out_path', type=str, default='../data/')
    parser.add_argument('--coco_path', type=str, default='/media/abiten/4TB/Datasets/COCO_raw/annotations/')
    parser.add_argument('--coco_split', type=str, default='/media/abiten/SSD-DATA/self-critical.pytorch/data/cocotalk.json')
    parser.add_argument('--word_model', type=str, default='/media/abiten/SSD-DATA/RelationalAxioms/data/fasttext/cc.en.300.bin' )
    args = parser.parse_args()

    coco_train = COCO(os.path.join(args.coco_path, 'instances_train2014.json'))
    coco_val = COCO(os.path.join(args.coco_path, 'instances_val2014.json'))
    coco_split = json.load(open(args.coco_split, 'r'))

    create_folder(args.out_path, 'fasttext_att')
    create_folder(args.out_path, 'fasttext_box')
    create_folder(args.out_path, 'obj_ids')

    # Get the fasttext representation
    if not os.path.exists(os.path.join(args.out_path, 'objects_word_repr')):
        model = fasttext.load_model(args.word_model)
        word_repr = {}
        for elm in coco_train.cats.values():
            word_repr[elm['id']] = model.get_word_vector(elm['name'])
        pickle.dump(word_repr, open(os.path.join(args.out_path, 'objects_word_repr'), 'wb'))

        # Just in case
        obj_names = {}
        for elm in coco_train.cats.values():
            obj_names[elm['name']] = elm['id']
        pickle.dump(obj_names, open(os.path.join(args.out_path, 'object_names'), 'wb'))

    else:
        word_repr = pickle.load(open(os.path.join(args.out_path, 'objects_word_repr'), 'rb'))

    bad_images = 0
    for elm in tqdm.tqdm(coco_split['images']):
        obj = []
        obj_ids = []
        box = []
        img_id = elm['id']
        if elm['split'] == 'train':
            anns = coco_train.imgToAnns[img_id]
        else:
            anns = coco_val.imgToAnns[img_id]

        for ann in anns:
            obj_ids.append(ann['category_id'])
            obj.append(word_repr[ann['category_id']])
            # Convert bbox info format from x1,y1,w,h to x1,y1,x2,y2
            bbox = ann['bbox']
            bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            bbox = [int(b) for b in bbox]
            box.append(bbox)
        if obj != []:
            # This is extremely inefficient but I am lazy and this works!!
            np.savez_compressed(os.path.join(args.out_path, 'fasttext_att', str(img_id)), feat=obj)
            np.save(os.path.join(args.out_path, 'fasttext_box', str(img_id)), box)
            np.save(os.path.join(args.out_path, 'obj_ids', str(img_id)), obj_ids)
        else:
            bad_images+=1
            # print('WTF!')

    print('Number of bad images: {}'.format(bad_images))
    # json.dump(new_coco_split, open(os.path.join(args.out_path, 'cocotalk_ft.json'), 'w'))