import json
import constant

def make_raw_metadata():
        """
    metadata 각 파일들에서 필요한 정보만 모은 리스트를 리턴한다.
    리스트의 index 는 파일 이름과 동일함 (즉, 0에는 dummy data)
    * example of each item_metadata format
    {
        "TOTAL": 2,
        "DATA": {
            "B01DDF6WWS": {
                "quantity": 1,
                "name": "iPhone SE Case, araree\u00ae [Airfit] Ultra Slim SOFT-Interior Scratch Protection with Perfect Fit for iPhone SE, 5S and 5 (2016) (BLUE(Matt))"
            },
            "B019775SYE": {
                "quantity": 1,
                "name": "Midline Lacrosse Logo Crew Socks (Columbia Blue/Navy, Small)"
            }
        }
    }
    :return: The list of item_metadata
    """
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

make_metadata(raw_metadata, valid_images):
    metadata = {}
    for i in valid_images:
        data = raw_metadata[i]
        if data['TOTAL'] > 0:
            if len(data['DATA'].keys()) == 1:
                for asin in data['DATA'].keys():
                    if asin in metadata:
                        metadata[asin]['repeat'] = d[asin]['repeat'] + 1
                        metadata[asin]['bin_list'].append(i)
                    else:
                        metadata[asin]['repeat'] = 1
                        metadata[asin]['bin_list'] = [i]
    return metadata

make_target_vector_map(metadata, isClustering):
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
    raw_metadata = make_raw_metadata()

    valid_images = range(1, len(raw_metadata))

    metadata = make_metadata(raw_metadata, valid_images)

    valid_object = []

    for asin in metadata.keys():
        if 20 <= d[asin]['repeat'] <= 70:
            valid_object.append(asin)

    valid_images = []

    for asin in valid_object:
        for i in metadata[asin]['bin_list']:
            valid_images.append(i)

    metadata = make_metadata(raw_metadata, valid_images)

    asin_index_map, index_asin_map = make_target_vector_map(metadata)


    with open(METADATA_FILE, 'w') as metadata_file:
        json.dump(metadata, metadata_file)

    with open(ASIN_INDEX_FILE, 'w') as asin_index_file:
        json.dump(asin_index_map, asin_index_file)
    with open(INDEX_ASIN_FILE, 'w') as index_asin_file:
        json.dump(index_asin_map, index_asin_file)

    with open(VALID_IMAGES_FILE, 'w') as valid_images_file:
        json.dump(valid_images, valid_images_file)

    print(len(valid_images))
    print(len(asin_index_map))
    print(len(valid_images)/len(asin_index_map))
