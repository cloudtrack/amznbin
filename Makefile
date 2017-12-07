demo :
	sh demo.sh

demo_count_LENET :
	python3 src/main.py --mode 'test' --model "LENET" --function "count" --difficulty "moderate" --batch 10

demo_count_VGGNET :
	python3 src/main.py --mode 'test' --model "VGGNET" --function "count" --difficulty "moderate" --batch 10

prepare_train : mini_process remove_invalid make_tfrecords

image_dataset : src/load_and_process_images.py
	python3 src/load_and_process_images.py

load_metadata :
	aws s3 cp s3://aft-vbi-pds/metadata/ dataset/metadata/ --recursive
	rm dataset/metadata/?.json
	rm dataset/metadata/??.json
	rm dataset/metadata/???.json
	rm dataset/metadata/????.json

remove_invalid : src/remove_invalid_images.py
	python3 src/remove_invalid_images.py

process_metadata : src/process_metadata.py
	python3 src/process_metadata.py

make_tfrecords : src/write_tfrecord.py
	python3 src/write_tfrecord.py

launch_image_cleaner : src/image_cleaner.py
	python3 src/image_cleaner.py

mini_process : src/mini_process.py
	python3 src/mini_process.py

reset_tfrecords :
	rm dataset/*.tfrecords
