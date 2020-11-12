path=/home/mrh/Record/tum_dataset/seq11/images
#path=/home/mrh/Record/cgdso_test

build/bin/create_voc_step0 \
sift \
voc/cgdso.sift \
${path}/*.jpg
