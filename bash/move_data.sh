#!/bin/bash
# move the data from the old folder train, test and valid dir to a single dir

# for logging to file
#exec &> move_rename_data.txt

File_Extension=*.jpg

for ds in "BSR" "CIFAR_10" "CIFAR_100";do
	target_dir=/home/lukhetho/Documents/MSC/Data/processed_data/${ds}/all_data/
	mkdir -p ${target_dir}
	count=1
	echo
	for split in "TRAIN" "TEST" "VAL"; do
		original_dir=/home/lukhetho/Documents/MSC/Data/FINAL_DATA/HETEROGENEOUS/${ds}/${split}/
		for fp in ${original_dir}${File_Extension};do
			echo "Processing: ${fp} ${ds}-${split} ======> Renamed: ${count}.jpg"

			cp  ${fp} $target_dir${count}.jpg
			count=`expr $count + 1`
		done
	done
done


File_Extension=*.png
for ds in  "WEIZMANN";do
	target_dir=/home/lukhetho/Documents/MSC/Data/processed_data/${ds}/all_data/
	mkdir -p ${target_dir}
	count=1

	for split in "TRAIN" "TEST" "VAL"; do
		original_dir=/home/lukhetho/Documents/MSC/Data/FINAL_DATA/HETEROGENEOUS/${ds}/${split}/
		for fp in ${original_dir}${File_Extension};do
			echo "Processing: ${fp} ${ds}-${split} ======> Renamed: ${count}.jpg"

			cp  ${fp} $target_dir${count}.jpg
			count=`expr $count + 1`
		done
	done
done



		
	
	


