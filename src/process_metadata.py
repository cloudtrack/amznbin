import json
import os
import os.path

from dataset import TOTAL_DATA_SIZE

METADATA_DIR = "dataset/metadata/"


# getting whole metadata list
def get_metadata():
    metadata = []
    for i in range(TOTAL_DATA_SIZE):
        if i % 1000 == 0:
            print("get_metadata: processing (%d/%d)..." % (i, TOTAL_DATA_SIZE))
        json_path = '%s%05d.json' % (METADATA_DIR, i+1)
        if os.path.isfile(json_path):
            d = json.loads(open(json_path).read())
            metadata.append(d)
    return metadata


def get_instance_data(metadata):
    instances = {}
    N = len(metadata)
    for i in range(TOTAL_DATA_SIZE):
        if i % 1000 == 0:
            print("get_instance_data: processing (%d/%d)..." % (i, N))
        if metadata[i]:
            quantity = metadata[i]['EXPECTED_QUANTITY']
            if quantity > 0:
                bin_info = metadata[i]['BIN_FCSKU_DATA']
                bin_keys = bin_info.keys()
                for j in range(0, len(bin_info)):
                    instance_info = bin_info[bin_keys[j]]
                    asin = instance_info['asin']
                    if asin in instances:
                        # occurance
                        instances[asin]['repeat'] = instances[asin]['repeat'] + 1
                        # quantity
                        instances[asin]['quantity'] = instances[asin]['quantity'] + instance_info['quantity']
                        instances[asin]['bin_list'].append(i)
                    else:
                        instances[asin] = {}
                        instances[asin]['repeat'] = 1
                        instances[asin]['quantity'] = instance_info['quantity']
                        instances[asin]['name'] = instance_info['name']
                        bin_list = list()
                        bin_list.append(i)
                        instances[asin]['bin_list'] = bin_list
    return instances


if __name__ == '__main__':
    metadata = get_metadata()
    # dumping out all metadata into a file
    print("dumping metadata.json...")
    with open('dataset/metadata.json', 'w') as fp:
        json.dump(metadata, fp)
    instances = get_instance_data(metadata)

    print("dumping instances.json...")
    with open('dataset/instances.json', 'w') as fp:
        json.dump(instances, fp)
