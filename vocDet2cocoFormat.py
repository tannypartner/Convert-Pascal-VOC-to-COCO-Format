# coding:utf-8

from PIL import Image, ImageDraw
from datetime import datetime
from copy import deepcopy
from shutil import move, rmtree
from pycocotools.coco import COCO
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)

import os
import time
import json
import random
import torchvision
import argparse

class VOC2COCO(object):
    def __init__(self, args):
        super().__init__()
        pp.pprint(f'{datetime.now()} VOC2COCO init...')
        self.year = str(args.year)

        # path
        self.root = args.output_dir_path # output dir path
        assert not os.path.exists(self.root), 'exist output-dir-path. Please chenge path.'
        os.makedirs(self.root)
        self.tmp_path = args.temporary_path if args.temporary_path else os.path.join(self.root, 'tmp') # download dataset path
        self.images_dir_path = os.path.join(self.root, 'images') # images dir path

        # download voc dataset
        pp.pprint(f'{datetime.now()} Download dataset...')
        self.train = torchvision.datasets.VOCDetection(self.tmp_path, year=self.year, image_set='train', download=True)
        self.val = torchvision.datasets.VOCDetection(self.tmp_path, year=self.year, image_set='val', download=True)
        self.trainval = torchvision.datasets.VOCDetection(self.tmp_path, year=self.year, image_set='trainval', download=True)
        self.test = torchvision.datasets.VOCDetection(self.tmp_path, year=self.year, image_set='test', download=True)

        # init coco dict
        self.cocoDict = {'train':{}, 'val':{}, 'trainval':{}, 'test':{}}
        # coco info
        info = {
            'description': f'voc{self.year} dataset',
            'url': f'torchvision.datasets.VOCDetection (torchvision.__version__:{torchvision.__version__})',
            'version': 1.0,
            'year': int(self.year),
            'date_created': datetime.today().strftime('%Y/%m/%d'),
        }
        # coco category (http://host.robots.ox.ac.uk/pascal/VOC/voc2007/dbstats.html)
        categories = ['background']
        categories += ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
        'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
        self.category_dict = {name:id for id,name in enumerate(categories, start=0)}
        # write (supercategory == name)
        for mode in self.cocoDict.keys():
            self.cocoDict[mode]['info'] = info
            self.cocoDict[mode]['categories'] = [
                {'id':i, 'supercategory':c, 'name':c} for c,i in self.category_dict.items()
            ]
            self.cocoDict[mode]['images'] = []
            self.cocoDict[mode]['annotations'] = []
            self.cocoDict[mode]['licenses'] = []

    def __call__(self):
        '''run'''
        self.__make()
        self.__write()
        self.__check_pycocotools()
        pp.pprint(f'{datetime.now()} finish!!!')

    def __make(self):
        '''make coco format'''
        # start dataset mode
        for dataset in [self.train, self.val, self.trainval, self.test]:
            mode = dataset.image_set
            annotation_id = 0
            pp.pprint(f'{datetime.now()} start {mode}...')

            # start dataset
            for _, item in dataset:

                # image
                image = dict()
                image['id'] = int(item['annotation']['filename'].split('.')[0])
                image['file_name'] = item['annotation']['filename']
                image['coco_url'] = os.path.join(os.path.basename(self.images_dir_path), image['file_name'])
                image['height'] = item['annotation']['size']['height']
                image['width'] = item['annotation']['size']['width']
                # pp.pprint(f'{datetime.now()} image id {image["id"]:06d}')

                # annotation
                for obj in item['annotation']['object']:
                    annotation = dict()
                    annotation['iscrowd'] = 0
                    annotation['image_id'] = image['id']
                    x1 = float(obj['bndbox']['xmin']) - 1
                    y1 = float(obj['bndbox']['ymin']) - 1
                    x2 = float(obj['bndbox']['xmax']) - x1 # width
                    y2 = float(obj['bndbox']['ymax']) - y1 # height
                    annotation['bbox'] = [x1, y1, x2, y2] # x, y, width, height
                    annotation['area'] = float(x2 * y2) # width * height
                    annotation['category_id'] = self.category_dict[obj['name']]
                    annotation['id'] = annotation_id
                    annotation_id +=1

                    # add annotation in coco dict
                    self.cocoDict[mode]['annotations'].append(annotation)
                # add image in coco dict
                self.cocoDict[mode]['images'].append(image)

    def __write(self):
        '''write instance json files'''
        # start dataset mode
        for dataset in [self.train, self.val, self.trainval, self.test]:
            mode = dataset.image_set
            pp.pprint(f'{datetime.now()} [{mode}] write json & move images...')
            # write json
            with open(os.path.join(self.root, f'instances_{mode}.json'), 'w') as fw:
                json.dump(self.cocoDict[mode], fw, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))

        # move images
        move(os.path.join(self.tmp_path, 'VOCdevkit', f'VOC{self.year}', 'JPEGImages'), self.images_dir_path)
        # remove tmp dir (VOCdevkit)
        rmtree(os.path.join(self.tmp_path, 'VOCdevkit'))

    def __check_pycocotools(self):
        '''check pycocotools'''
        rc = lambda: random.randint(0, 255)
        cmap = {id:(rc(), rc(), rc()) for id in self.category_dict.values()}

        for mode in ['trainval', 'test']:
            pp.pprint(f'{datetime.now()} check start [{mode}]')
            cocoGt = COCO(os.path.join(self.root, f'instances_{mode}.json'))
            cats = cocoGt.loadCats(cocoGt.getCatIds())

            # check category
            assert [cat['name'] for cat in cats] == list(self.category_dict.keys()), 'check category'
            assert [cat['supercategory'] for cat in cats] == list(self.category_dict.keys()), 'check super category'

            # inference
            imagesGt_path = os.path.join(self.root, f'imagesGt_{mode}')
            os.makedirs(imagesGt_path)
            for image_info in cocoGt.loadImgs(cocoGt.getImgIds()):
                image_id = image_info['id']
                img = Image.open(os.path.join(self.root, image_info['coco_url']))
                draw = ImageDraw.Draw(img)
                for annotation in cocoGt.loadAnns(cocoGt.getAnnIds(imgIds=[image_id])):
                    (x, y, w, h) = list(map(int, annotation['bbox']))
                    draw.rectangle((x,y,x+w,y+h), fill=None, outline=cmap[annotation['category_id']], width=2)
                    draw.text((x,y), cocoGt.loadCats(cocoGt.getCatIds(catIds=[annotation['category_id']]))[0]['name'],
                              fill=cmap[annotation['category_id']], )
                img.save(os.path.join(imagesGt_path, image_info['file_name']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir-path', '-o', type=str, required=True,
                        help='output dir')
    parser.add_argument('--year', '-y', default='2012', type=int,
                        help='The dataset year, supports years 2007 to 2012. (defualt:2012)')
    parser.add_argument('--temporary-path', '-t', default=None, type=str,
                        help='download datasets path. (default:output-dir-path/tmp)')
    instance = VOC2COCO(args=parser.parse_args())
    instance()
