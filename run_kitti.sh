path=/home/mrh/Record/cgdso_test
kitti_path=/media/mrh/Media/ubuntuRecord/KITTI
no_of_dataser=05

build/bin/dso_kitti \
files=${kitti_path}/data_odometry_gray/sequences/${no_of_dataser} \
calib=${path}/camera_kitti.txt \
gamma=${path}/pcalib.txt \
vignette=${path}/vignette_kitti.png \
platformSettingPath=/home/mrh/catkin_ws/src/cgdso/config/kitti_setting.xml \
preset=2 \
mode=1 
