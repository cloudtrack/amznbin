import json
from constants import TOTAL_DATA_SIZE, METADATA_DIR, RAW_METADATA_FILE, METADATA_FILE, ASIN_INDEX_FILE, INDEX_ASIN_FILE, VALID_IMAGES_FILE

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

def make_metadata(raw_metadata, valid_images):
    metadata = {}
    for i in valid_images:
        data = raw_metadata[i]
        if data['TOTAL'] > 0:
            if len(data['DATA'].keys()) == 1:
                for asin in data['DATA'].keys():
                    if asin in metadata:
                        metadata[asin]['repeat'] = metadata[asin]['repeat'] + 1
                        metadata[asin]['bin_list'].append(i)
                    else:
                        metadata[asin] = {}
                        metadata[asin]['repeat'] = 1
                        metadata[asin]['bin_list'] = [i]
    return metadata

def make_target_vector_map(metadata):
    asin_index_map = {}
    index_asin_map = {}
    index = 0
    for asin in metadata.keys():
        if asin not in asin_index_map.keys():
            asin_index_map[asin] = index
            index_asin_map[index] = asin
            index += 1
    return asin_index_map, index_asin_map


if __name__ == '__main__':
    with open(RAW_METADATA_FILE, 'r') as raw_metadata_file:
        raw_metadata = json.load(raw_metadata_file)

    valid_images = range(1, len(raw_metadata))

    metadata = make_metadata(raw_metadata, valid_images)

    valid_object = []

    repeat_num = int(input())

    for asin in metadata.keys():
        if repeat_num <= metadata[asin]['repeat']:
            valid_object.append(asin)

    valid_images = []

    for asin in valid_object:
        for i in metadata[asin]['bin_list']:
            valid_images.append(i)

    metadata = make_metadata(raw_metadata, valid_images)

    asin_index_map, index_asin_map = make_target_vector_map(metadata)

    vl = len(valid_images)

    zero_cnt = 0
    for i in range(1, len(raw_metadata)):
        if raw_metadata[i]['TOTAL'] == 0:
            valid_images.append(i)
            zero_cnt = zero_cnt + 1
            if zero_cnt * 10 >= vl:
                break


    with open(METADATA_FILE, 'w') as metadata_file:
        json.dump(metadata, metadata_file)

    with open(ASIN_INDEX_FILE, 'w') as asin_index_file:
        json.dump(asin_index_map, asin_index_file)
    with open(INDEX_ASIN_FILE, 'w') as index_asin_file:
        json.dump(index_asin_map, index_asin_file)

    with open(VALID_IMAGES_FILE, 'w') as valid_images_file:
        json.dump(sorted(valid_images), valid_images_file, indent=4)

    print(len(valid_images))
    print(len(asin_index_map))
    print(len(valid_images)/(len(asin_index_map) + 1))
