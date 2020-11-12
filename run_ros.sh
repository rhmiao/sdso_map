path=/home/mrh/Record/cgdso_test

build/bin/dso_ros \
files=${path}/2018_05_29.zip \
calib=${path}/camera.txt \
gamma=${path}/pcalib.txt \
vignette=${path}/vignette.png \
platformSettingPath=/home/mrh/catkin_ws/src/cgdso/config/euroc_setting.xml \
preset=1 \
mode=1
