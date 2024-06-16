# 2MAF
2M-AF: A Strong Multi-Modality Framework For Human Action Quality Assessment with 61 Self-supervised Representation Learning

## Datasets

### AQA-7

Request dataset：http://rtis.oit.unlv.edu/datasets.html

### UNLV-diving

Request dataset: ：http://rtis.oit.unlv.edu/datasets.html

### MMFS-63

Request dataset: https://github.com/dingyn-reno

## data process

We use mmaction to extract skeleton points, reference: https://mmaction2.readthedocs.io

## Train/eval

```shell
python run.py --config config/AQA7.yaml --device 0
```



