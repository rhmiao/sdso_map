path=/home/mrh/Record/tum_dataset/seq11

build/bin/dso_dataset \
files=${path}/images.zip \
calib=${path}/camera.txt \
gamma=${path}/pcalib.txt \
vignette=${path}/vignette.png \
preset=0 \
mode=0
