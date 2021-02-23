#!/bin/bash
# ======== This bash script is used to rename the images numerically ========= #
# 'WEIZMANN' 'BSR'  'COCCO' 'CIFAR_100' 'CIFAR_10'
dataset="CIFAR_10"
File_Extension=*.jpg
# Target folder to rename files.
Target_directory="../../Data/processed_data/$dataset/inputs/"
# Destination folder to save renamed files.
Destination_directory="../../Data/processed_data/$dataset/images/"

# mkdir -p $Destination_directory
count=1
for files in $Target_directory$File_Extension
    do
    # Extension of the files.
    echo "Processing: ${files} ======> Renamed: ${count}.jpg"
    
    cp   "${files}"  "$Destination_directory/${count}.jpg"
    count=`expr $count + 1`
done
