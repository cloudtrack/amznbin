import glob
import os

import boto3
from PIL import Image

from constants import IMAGE_SIZE, IMAGE_DIR
from dataset import TOTAL_DATA_SIZE


def connect_s3_bucket():
    s3 = boto3.resource('s3', region_name='us-east-1')
    bucket = s3.Bucket(name='aft-vbi-pds')
    return bucket


def process_image(image_file):
    image = Image.open(image_file).convert('RGB')
    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LINEAR)
    resized_image.save(image_file)


if __name__ == '__main__':
    if not os.path.exists(IMAGE_DIR):
        print('mkdir ' + IMAGE_DIR)
        os.makedirs(IMAGE_DIR)
    start = len(glob.glob(IMAGE_DIR + '/*.jpg')) if len(glob.glob(IMAGE_DIR + '/*.jpg')) > 0 else 1
    print('Start from %05d.jpg' % start)
    bucket = connect_s3_bucket()
    for i in range(start, TOTAL_DATA_SIZE + 1):
        filename = '%05d.jpg' % i
        source = 'bin-images/' + filename
        dest = IMAGE_DIR + filename
        bucket.download_file(source, dest)
        process_image(dest)
        print('Processed {0}({1}/{2})'.format(dest, i, TOTAL_DATA_SIZE))

    print('Processing Done')
