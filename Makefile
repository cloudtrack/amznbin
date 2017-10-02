demo :
	sh demo.sh
image_dataset : src/load_and_process_images.py
	python3 src/load_and_process_images.py
meta_dataset :
	aws s3 cp s3://aft-vbi-pds/metadata/ dataset/metadata/ --recursive
	ls dataset/metadata | grep -P "^[\d]{1,4}\.[\w]+" | xargs -d"\n" rm
