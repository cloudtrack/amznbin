import json

from constants import METADATA_DIR, RAW_METADATA_FILE, METADATA_FILE, TOTAL_DATA_SIZE, ASIN_INDEX_FILE, INDEX_ASIN_FILE, MINIMUM_REPEAT


def make_raw_metadata():
    raw_metadata = [{}]
    for i in range(1, TOTAL_DATA_SIZE+1):
        if i % 1000 == 0:
            print("make_raw_metadata - processing (%d/%d)..." % (i, TOTAL_DATA_SIZE))
        file_path = '%s%05d.json' % (METADATA_DIR, i)
        json_data = json.loads(open(file_path).read())
        processed_json_data = {
            'TOTAL': json_data['EXPECTED_QUANTITY'],
            'DATA': {}
        }
        for bin_key in json_data['BIN_FCSKU_DATA'].keys():
            bin_meta = json_data['BIN_FCSKU_DATA'][bin_key]
            useful_meta = {
                'name': bin_meta['name'],
                'quantity': bin_meta['quantity'],
            }
            processed_json_data['DATA'][bin_key] = useful_meta
        raw_metadata.append(processed_json_data)

    return raw_metadata


def make_metadata(raw_metadata):
    metadata = {}
    for i in range(1, len(raw_metadata)):
        if i % 1000 == 0:
            print("make_metadata - processing (%d/%d)..." % (i, TOTAL_DATA_SIZE))
        if raw_metadata[i]:
            quantity = raw_metadata[i]['TOTAL']
            if quantity > 0:
                bin_info = raw_metadata[i]['DATA']
                for bin_key in bin_info.keys():
                    instance_info = bin_info[bin_key]
                    asin = bin_key
                    if asin in metadata:
                        # occurance
                        metadata[asin]['repeat'] = metadata[asin]['repeat'] + 1
                        # quantity
                        metadata[asin]['quantity'] = metadata[asin]['quantity'] + instance_info['quantity']
                        metadata[asin]['bin_list'].append(i)
                    else:
                        metadata[asin] = {}
                        metadata[asin]['repeat'] = 1
                        metadata[asin]['quantity'] = instance_info['quantity']
                        metadata[asin]['name'] = instance_info['name']
                        metadata[asin]['bin_list'] = [i]
    return metadata


def make_target_vector_map(metadata):
    asin_index_map = {}
    index_asin_map = {}
    index = 0
    i = 0
    for asin in metadata.keys():
        if i % 1000 == 0:
            print("make_target_vector_map - processing (%d/%d)..." % (i, len(metadata.keys())))
        i += 1
        if metadata[asin]['repeat'] >= MINIMUM_REPEAT:
            if asin not in asin_index_map.keys():
                asin_index_map[asin] = index
                index_asin_map[index] = asin
                index += 1
    print(str(index))
    return asin_index_map, index_asin_map

def classify_images(asin_index_map, raw_metadata):
    valid_images = []
    invalid_images = []
    for i in range(1, len(raw_metadata)):
        if raw_metadata[i]:
            bin_info = raw_metadata[i]['DATA']
            flag = True
            for bin_key in bin_info.keys():
                bin_index = asin_index_map.get(bin_key)
                if not bin_index:
                    flag = False
            if flag:
                valid_images.append('%05d' % i)
            else:
                invalid_images.append('%05d' % i)
    return valid_images, invalid_images



if __name__ == '__main__':
    raw_metadata = make_raw_metadata()
    metadata = make_metadata(raw_metadata)
    asin_index_map, index_asin_map = make_target_vector_map(metadata)
    valid_images, invalid_images = count_images(asin_index_map, raw_metadata)
    print (str(len(valid_images))+" "+str(len(invalid_images)))
