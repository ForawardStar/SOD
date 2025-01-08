## Salient Object Detection via Adaptive Boundary-Location Feature Fusion

Salient object detection aims to identify objects in natural images that most catch human attention. A common approach in this field involves integrating diverse features learned from multiple network branches. Intuitively, to enhance accuracy, the fusion process shall adapt to individual inputs with varying patterns. To improve such adaptability, this project develops a boundary-location information fusion framework. Specifically, we decompose the original problem into two simplified sub-tasks: object localization and boundary refinement, each tackled by a dedicated branch. To mitigate the negative impact of misleading attributions in the feature space, we further propose an adaptive feature augmentation technique, which dynamically pinpoints potential attributions causing incorrect predictions, thereby enhancing robustness. Additionally, an adaptive cross-attention module is devised with parameters that adjust dynamically to effectively aggregate boundary and location features. 


If you run into any problems or feel any difficulties to run this code, do not hesitate to leave issues in this repository.

My e-mail is: yuanbinfu@tju.edu.cn

This repository contains:

- [x] Full code, data for `training` and `testing`
- [x] Pretrained models based on [P2T](https://arxiv.org/abs/2106.12011) and MobileNetV2
- [x] Fast preparation script (based on github release)


## Environment Installation
Our model is based on the following packages:
* python 3.6+
* pytorch >=1.6, torchvision, OpenCV-Python, tqdm
* Tested on pyTorch 1.7.1

They can be installed following the instructions of the official website: https://pytorch.org/

Or, you can install the packages using requirement.txt, through running:
```pip install -r requirement.txt```

## Data Preparing

**You can choose to use our automatic preparation script, if you have good downloading speed on github**:
```
bash scripts/prepare_data.sh
```
The script will prepare the training and testing datasets. 
If you suffer from slow downloading rate and luckily you have a proxy, a powerful tool [Proxychains4](https://github.com/rofl0r/proxychains-ng) can help you to execute the script through your own proxy by running the following command: `proxychains4 bash scripts/prepare_data.sh`.

If you have a low downloading speed, please download the training data manually: 

* Preprocessed data of 6 datasets: [[Google Drive]](https://drive.google.com/file/d/1fj1KoLa8uOBmGMkpKkjj7xVHciSd8_4V/view?usp=sharing), [[Baidu Pan, ew9i]](https://pan.baidu.com/s/1tNGQS9SjFu9hm0a0svnlvg?pwd=ew9i)

We have processed the data well so you can use them without any preprocessing steps. 
After completion of downloading, extract the data and put them to `./data/` folder:

```
unzip SOD_datasets.zip -O ./data
```


## Demo

We provide some examples for quick run:
````
python demo.py
````

## Train

If you cannot run `bash scripts/prepare_data.sh`, please first download the imagenet pretrained models and put them to `pretrained` folder:

* [[Google Drive]](https://drive.google.com/drive/folders/1ios0nOHQt61vsmu-pdkpS1zBb_CwLrmk?usp=sharing), [[Baidu Pan,eae8]](https://pan.baidu.com/s/1xJNJ8SEDwKMHxlFh3yCUeQ?pwd=eae8)


It is very simple to train our network. We have prepared a script to train our model:
```
bash ./scripts/train.sh
```

To train our model, you need to change the params in `scripts/train.sh`. Please refer to the comments in the last part of `scripts/train.sh` for more details (very simple).

### Test

#### Pretrained Models

Download them from the following urls if you did not run `bash scripts/prepare_data.sh` to prepare the data:

* [[Google Drive]](https://drive.google.com/drive/folders/1Un6trEOTIVza2wH5Q2PAQVNGgsKEEHv4?usp=sharing), [[Baidu Pan,eae8]](https://pan.baidu.com/s/1xJNJ8SEDwKMHxlFh3yCUeQ?pwd=eae8)

#### Generate Saliency Maps

After preparing the pretrained models, it is also very simple to generate saliency maps via EDN-VGG16/EDN-R50/EDN-Lite/EDN-LiteEX:

```
bash ./tools/test.sh
```

The scripts will automatically generate saliency maps on the `salmaps/` directory.

* For computing Fbw, S-m, and E-m measures, please use the official MATLAB code to generate the results: [Download Code Here](https://github.com/yuhuan-wu/EDN/files/13497335/fbw-sm-em.zip).

#### Pre-computed Saliency maps

For covenience, we provide the pretrained saliency maps on several datasets by:

* Running the command `bash scripts/prepare_salmaps.sh` to download them to `salmaps` folder.
* Or downloading them manually: [[Google Drive]](https://drive.google.com/drive/folders/1MymUy-aZx_45YJSOPd3GQjwel-YBTUPX?usp=sharing), [[Baidu Pan, c9zm]](https://pan.baidu.com/s/1HAZTrJhIkw8JdACN_ChGWA?pwd=c9zm)
* Now we have included all saliency maps of EDN varies, including EDN-VGG16, EDN-ResNet-50, **EDN-P2T-Small**, EDN-Lite, and EDN-LiteEX.

#### Contact

* I encourage everyone to contact me via my e-mail. My e-mail is: yuanbinfu@tju.edu.cn

