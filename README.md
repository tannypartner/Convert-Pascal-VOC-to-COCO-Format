# Convert-Pascal-VOC-to-COCO-Format

The python script for converting Pascal VOC dataset provided in
torchvision.datasets to coco format.

## note

**Please note that only object detection is supported.**

## Preparation

```python
# install requirements
pip install pillow pycocotools torchvision
```

## Demo

### 1.Standard

```python
# run
python vocDet2cocoFormat.py -o ./datasets/VOC_coco_format
```

### 2.If you have previously downloaded the VOC dataset

```shell
# The structure of dir

├── datasets
│   └── VOC2007                         # Pascal VOC 2007 image set
│       ├── VOCtest_06-Nov-2007.tar     # test image set
│       └── VOCtrainval_06-Nov-2007.tar # train or validation image set
├── vocDet2cocoFormat.py                # python script
├── LICENSE
└── README.md
```

```python
# run
python vocDet2cocoFormat.py -o ./datasets/VOC_coco_format -t ./datasets/VOC2007 -y 2007
```

## Help

```shell
usage: vocDet2cocoFormat.py [-h] --output-dir-path OUTPUT_DIR_PATH
                            [--year YEAR] [--temporary-path TEMPORARY_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --output-dir-path OUTPUT_DIR_PATH, -o OUTPUT_DIR_PATH
                        output dir
  --year YEAR, -y YEAR  The dataset year, supports years 2007 to 2012.
                        (defualt:2012)
  --temporary-path TEMPORARY_PATH, -t TEMPORARY_PATH
                        download datasets path. (default:output-dir-path/tmp)
```

## Updata Log

[2020-09-07] upload script for object detection.

[2020-09-07] create this repository.

## Reference

- [torchvision.datasets — PyTorch 1.6.0 documentation](https://pytorch.org/docs/stable/torchvision/datasets.html)
- [COCO Formatの作り方 - Qiita](https://qiita.com/harmegiddo/items/da131ae5bcddbbbde41f)
- [cocodataset/cocoapi: COCO API - GitHub](https://github.com/cocodataset/cocoapi)

## License

MIT License
