#!/bin/bash
# script for downloading the dataset

cd data
# the host was down, likely due to the AWS incident
#wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
#wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

# we will use this mirror for now
wget https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_06-Nov-2007.tar
wget https://data.brainchip.com/dataset-mirror/voc/VOCtest_06-Nov-2007.tar

tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar

cd ..
