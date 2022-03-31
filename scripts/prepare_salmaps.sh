set -o xtrace


BASE_ADDR=https://github.com/yuhuan-wu/EDN/releases/download/v1.0
EDN_VGG16_ADDR=${BASE_ADDR}/EDN-VGG16.zip
EDN_R50_ADDR=${BASE_ADDR}/EDN-R50.zip
EDN_LITE_ADDR=${BASE_ADDR}/EDN-Lite.zip
EDN_LITEEX_ADDR=${BASE_ADDR}/EDN-LiteEX.zip

# Download saliency maps to salmaps/
mkdir salmaps

wget -c --no-check-certificate --content-disposition $EDN_R50_ADDR -O salmaps/EDN-R50.zip
wget -c --no-check-certificate --content-disposition $EDN_VGG16_ADDR -O salmaps/EDN-VGG16.zip
wget -c --no-check-certificate --content-disposition $EDN_LITE_ADDR -O salmaps/EDN-Lite.zip
wget -c --no-check-certificate --content-disposition $EDN_LITEEX_ADDR -O salmaps/EDN-LiteEX.zip