set -o xtrace

BASE_ADDR=https://github.com/yuhuan-wu/EDN/releases/download/v1.0

DATA_ADDR=${BASE_ADDR}/SOD_datasets.zip
VGG16_MODEL_ADDR=${BASE_ADDR}/5stages_vgg16_bn-6c64b313.pth
EDN_VGG16_ADDR=${BASE_ADDR}/EDN-VGG16.pth
EDN_R50_ADDR=${BASE_ADDR}/EDN-R50.pth
EDN_LITE_ADDR=${BASE_ADDR}/EDN-Lite.pth
EDN_LITEEX_ADDR=${BASE_ADDR}/EDN-LiteEX.pth


# download data from github address
wget -c --no-check-certificate --content-disposition $DATA_ADDR -O SOD_datasets.zip
unzip -n SOD_datasets.zip -d data/

# download imagenet pretrained models
mkdir pretrained

echo "downloading imagenet pretrained resnet model from pytorch.org!"
wget -c --no-check-certificate --content-disposition https://download.pytorch.org/models/resnet50-19c8e357.pth -O pretrained/resnet50-19c8e357.pth

echo "downloading imagenet pretrained mobilenetv2 model from pytorch.org!"
wget -c --no-check-certificate --content-disposition https://download.pytorch.org/models/mobilenet_v2-b0353104.pth -O pretrained/mobilenet_v2-b0353104.pth

echo "downloading imagenet pretraiend vgg16 model from github.com!"
wget -c --no-check-certificate --content-disposition $VGG16_MODEL_ADDR -O pretrained/5stages_vgg16_bn-6c64b313.pth

# download pretrained model for salient object detection

echo "downloading resnet50-based EDN from github.com!"
wget -c --no-check-certificate --content-disposition $EDN_R50_ADDR -O pretrained/EDN-R50.pth

echo "downloading vgg16-based EDN from github.com!"
wget -c --no-check-certificate --content-disposition $EDN_VGG16_ADDR -O pretrained/EDN-VGG16.pth

echo "downloading EDNLite and EDN-LiteEX from github.com!"
wget -c --no-check-certificate --content-disposition $EDN_LITE_ADDR -O pretrained/EDN-Lite.pth
wget -c --no-check-certificate --content-disposition $EDN_LITEEX_ADDR -O pretrained/EDN-Lite.pth
