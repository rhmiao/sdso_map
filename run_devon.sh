path=/home/mrh/Record/cgdso_test
kitti_path=/media/mrh/Media/ubuntuRecord/devonIsland
no_of_dataser=s13

build/bin/dso_kitti \
files=${kitti_path}/${no_of_dataser} \
calib=${path}/camera_devon.txt \
gamma=${path}/pcalib.txt \
vignette=${path}/vignette_devon.png \
platformSettingPath=/home/mrh/catkin_ws/src/cgdso/config/devonIsland_setting.xml \
preset=0 \
mode=1 
