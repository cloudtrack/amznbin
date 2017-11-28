demo :
	sh demo.sh
image_dataset : src/load_and_process_images.py
	python3 src/load_and_process_images.py
load_metadata :
	aws s3 cp s3://aft-vbi-pds/metadata/ dataset/metadata/ --recursive
	rm dataset/metadata/?.json
	rm dataset/metadata/??.json
	rm dataset/metadata/???.json
	rm dataset/metadata/????.json
process_metadata : src/process_metadata.py
	python3 src/process_metadata.py
tf_record : src/write_tfrecord.py
	python3 src/write_tfrecord.py
