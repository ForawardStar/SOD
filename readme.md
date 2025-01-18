## Salient Object Detection via Adaptive Boundary-Location Feature Fusion

Salient object detection aims to identify objects in natural images that most catch human attention. A common approach in this field involves integrating diverse features learned from multiple network branches. Intuitively, to enhance accuracy, the fusion process shall adapt to individual inputs with varying patterns. To improve such adaptability, this project develops a boundary-location information fusion framework. Specifically, we decompose the original problem into two simplified sub-tasks: object localization and boundary refinement, each tackled by a dedicated branch. To mitigate the negative impact of misleading attributions in the feature space, we further propose an adaptive feature augmentation technique, which dynamically pinpoints potential attributions causing incorrect predictions, thereby enhancing robustness. Additionally, an adaptive cross-attention module is devised with parameters that adjust dynamically to effectively aggregate boundary and location features. 

---

If you run into any problems or feel any difficulties to run this code, do not hesitate to leave issues in this repository. We will release the training codes once the paper has been published.

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

* Preprocessed data of 5 datasets: [[Google Drive]](https://drive.google.com/file/d/1fj1KoLa8uOBmGMkpKkjj7xVHciSd8_4V/view?usp=sharing), [[Baidu Pan, ew9i]](https://pan.baidu.com/s/1tNGQS9SjFu9hm0a0svnlvg?pwd=ew9i)

We have processed the data well so you can use them without any preprocessing steps. 
After completion of downloading, extract the data and put them to `./data/` folder:

```
unzip SOD_datasets.zip -O ./data
```


## Demo

We provide some examples for quick inference. You should download our trained model from：https://pan.quark.cn/s/c4a28f09b1f2, and put it into the folder "checkpoints/".


## Acknowledgment

Our codes are built upon https://github.com/yuhuan-wu/EDN/
 Then, you can simply run:
````
python demo.py
````
The JPEG image files placed in the folder 'examples/' will be processed by our model, and the results will also be saved into the folder 'examples/' with the suffix '_Ours.png'. 


#### Contact

* I encourage everyone to contact me via my e-mail. My e-mail is: yuanbinfu@tju.edu.cn

