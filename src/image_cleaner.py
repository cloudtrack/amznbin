from dataset import load_dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json

def save (invalid, semi, checked) :
    print("dumping invalid_images.json")
    with open('dataset/invalid_images.json', 'w') as invalid_images_file:
        json.dump(invalid, invalid_images_file)
        print("%d invalid images" % len(invalid))

    print("dumping semi invalid_images.json")
    with open('dataset/semi_invalid_images.json', 'w') as semi_invalid_images_file:
        json.dump(semi, semi_invalid_images_file)
        print("%d semi invalid images" % len(semi))

    print("dumping checked_images.json")
    with open('dataset/checked_images.json', 'w') as checked_images_file:
        json.dump(checked, checked_images_file)
        print("%d checked images" % len(checked))

def clean (data, invalid, semi, checked) :
    image_tensor, image_index_tensor = data.get_batch_tensor(1)
    
    with tf.Session() as _sess:
        _sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=_sess, coord=coord)
        key_input = ""
        img_cnt = 1
        try:
            while (not coord.should_stop()) and (key_input != 'q'):
                image, indice = _sess.run([image_tensor, image_index_tensor])
                label = data.get_labels_from_indices(indice, 'count', 'moderate')

                if indice not in (checked + invalid + semi):
                    image = image[0]
                    indice = int(indice[0])

                    print('%dth image' % img_cnt)
                    print(indice)
                    print(np.argmax(label))

                    plt.ion()
                    plt.imshow(image, interpolation='nearest')
                    plt.axis('off')
                    plt.show()
                    
                    key_input = input()

                    if key_input == 'i' :
                        invalid += [indice]
                        plt.close()

                    elif key_input == 's' :
                        semi += [indice]
                        plt.close()

                    else:
                    	if key_input != 'q':
	                        checked += [indice]
	                        plt.close()

                    img_cnt = img_cnt + 1

        except tf.errors.OutOfRangeError:
            print('epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)

    save(list(invalid), list(semi), list(checked))
    total_images = json.loads(open("dataset/valid_images.json").read())

invalid = json.loads(open("dataset/invalid_images.json").read())
semi = json.loads(open("dataset/semi_invalid_images.json").read())
checked = json.loads(open("dataset/checked_images.json").read())

dataset = load_dataset()
train, valid, test = dataset.train, dataset.validation, dataset.test

# clean(train, invalid, semi, checked)
# clean(valid, invalid, semi, checked)
# clean(test, invalid, semi, checked)

