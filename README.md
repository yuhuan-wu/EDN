# EDN

[IEEE TIP 2022, EDN: Salient Object Detection via Extremely-Downsampled Network](https://arxiv.org/abs/2012.13093)


If you run into any problems or feel any difficulties to run this code, do not hesitate to leave issues in this repository.

My e-mail is: wuyuhuan @ mail.nankai (dot) edu.cn


This repository contains:

- [x] Full code, data, pretrained models for `training` and `testing`
- [x] Fast preparation script

### Requirements

* python 3.6+
* pytorch >=1.6, torchvision, OpenCV-Python, tqdm
* Tested on pyTorch 1.7.1

Simply using:
````
pip install -r requirements.txt
````
to install all requirements.

### Run all steps quickly

Simply run:

```
bash one-key-run.sh
```

It will download all data, evaluate all models, produce all saliency maps to `salmaps/` folder,  and train `EDN-Lite` automatically.


### Data Preparing

**You can choose to use our automatic preparation script, if you have good downloading speed on github**:
```
bash scripts/prepare_data.sh
```
The script will prepare the datasets, imagenet-pretrained models, and pretrained models of EDN/EDN-Lite.


If you have a low downloading speed, please download the training data manually: 

* Preprocessed data of 6 datasets: [[Google Drive]](https://drive.google.com/file/d/1fj1KoLa8uOBmGMkpKkjj7xVHciSd8_4V/view?usp=sharing), [[Baidu Pan]]()

We have processed the data well so you can use them without any preprocessing steps. 
After completion of downloading, extract the data and put them to `./data/` folder:

```
unzip SOD_datasets.zip -O ./data
```

### Demo

We provide some examples for quick run:
````
python demo.py
````

### Train

If you cannot run `bash scripts/prepare_data.sh`, please first download the imagenet pretrained models and put them to `pretrained` folder:

* [[Google Drive]](https://drive.google.com/drive/folders/1ios0nOHQt61vsmu-pdkpS1zBb_CwLrmk?usp=sharing), [[Baidu Pan,eae8]](https://pan.baidu.com/s/1xJNJ8SEDwKMHxlFh3yCUeQ?pwd=eae8)


It is very simple to train our network. We have prepared a script to train EDN-Lite:
```
bash ./scripts/train.sh
```

To train EDN-VGG16 or EDN-R50, you need to change the params in `scripts/train.sh`. Please refer to the comments in the last part of `scripts/train.sh` for more details (very simple).

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


#### Pre-computed Saliency maps

For covenience, we provide the pretrained saliency maps on several datasets by:

* Running the command `bash scripts/prepare_salmaps.sh` to download them to `salmaps` folder.
* Or downloading them manually: [[Google Drive]](https://drive.google.com/drive/folders/1MymUy-aZx_45YJSOPd3GQjwel-YBTUPX?usp=sharing), [[Baidu Pan, c9zm]](https://pan.baidu.com/s/1HAZTrJhIkw8JdACN_ChGWA?pwd=c9zm)



### Others 

#### TODO

1. Add results with the [P2T](https://arxiv.org/abs/2106.12011) transformer backbone.

#### Contact

* I encourage everyone to contact me via my e-mail. My e-mail is: wuyuhuan @ mail.nankai (dot) edu.cn

#### License

The code is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) for NonCommercial use only.


#### Citation

If you are using the code/model/data provided here in a publication, please consider citing our works:

````
@ARTICLE{wu2022edn,
  title={EDN: Salient object detection via extremely-downsampled network},
  author={Wu, Yu-Huan and Liu, Yun and Zhang, Le and Cheng, Ming-Ming and Ren, Bo},
  journal={IEEE Transactions on Image Processing},
  year={2022}
}

@ARTICLE{wu2021mobilesal,
  author={Wu, Yu-Huan and Liu, Yun and Xu, Jun and Bian, Jia-Wang and Gu, Yu-Chao and Cheng, Ming-Ming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={MobileSal: Extremely Efficient RGB-D Salient Object Detection}, 
  year={2021},
  doi={10.1109/TPAMI.2021.3134684}
}
````
