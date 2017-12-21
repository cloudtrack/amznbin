from PIL import Image
from constants import VALID_IMAGES_FILE, IMAGE_DIR
import json


with open(VALID_IMAGES_FILE, 'r') as valid_images_file:
    old_valid_images = json.load(valid_images_file)

valid_images = []

for i in old_valid_images:
    if i < 1000000:
        valid_images.append(i)

new_valid_images = []

cmd = int(input())

for i in valid_images:
    if i < 1000000:
        image = Image.open('%s%05d.jpg' % (IMAGE_DIR, i))
        if cmd >= 1:
            new_i = i + 1000000
            t_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            t_image.save('%s%05d.jpg' % (IMAGE_DIR, new_i))
            new_valid_images.append(new_i)
        if cmd >= 2:
            new_i = i + 2000000
            t_image = image.transpose(Image.FLIP_TOP_BOTTOM)
            t_image.save('%s%05d.jpg' % (IMAGE_DIR, new_i))
            new_valid_images.append(new_i)

            new_i = i + 3000000
            t_image = image.transpose(Image.ROTATE_180)
            t_image.save('%s%05d.jpg' % (IMAGE_DIR, new_i))
            new_valid_images.append(new_i)
        if cmd >= 3:
            new_i = i + 4000000
            t_image = image.transpose(Image.ROTATE_90)
            t_image.save('%s%05d.jpg' % (IMAGE_DIR, new_i))
            new_valid_images.append(new_i)

            new_i = i + 5000000
            t_image = image.transpose(Image.ROTATE_270)
            t_image.save('%s%05d.jpg' % (IMAGE_DIR, new_i))
            new_valid_images.append(new_i)

            new_i = i + 6000000
            t_image = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
            t_image.save('%s%05d.jpg' % (IMAGE_DIR, new_i))
            new_valid_images.append(new_i)

            new_i = i + 7000000
            t_image = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(IMAGE.ROTATE_270)
            t_image.save('%s%05d.jpg' % (IMAGE_DIR, new_i))
            new_valid_images.append(new_i)



for i in new_valid_images:
    valid_images.append(new_valid_images)

with open(VALID_IMAGES_FILE, 'w') as valid_images_file:
    json.dump(sorted(valid_images), valid_images_file, indent=4)

