import json

from constants import METADATA_DIR, RAW_METADATA_FILE, METADATA_FILE, TOTAL_DATA_SIZE, ASIN_INDEX_FILE, INDEX_ASIN_FILE


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


def make_target_vector_map(raw_metadata):
    asin_index_map = {}
    index_asin_map = {}
    index = 0
    for i in range(1, len(raw_metadata)):
        if i % 1000 == 0:
            print("make_target_vector_map - processing (%d/%d)..." % (i, TOTAL_DATA_SIZE))
        data = raw_metadata[i]
        for asin in data['DATA'].keys():
            if asin not in asin_index_map.keys():
                asin_index_map[asin] = index
                index_asin_map[index] = asin
                index += 1

    return asin_index_map, index_asin_map


if __name__ == '__main__':
    raw_metadata = make_raw_metadata()
    print("dumping " + RAW_METADATA_FILE)
    with open(RAW_METADATA_FILE, 'w') as raw_metadata_file:
        json.dump(raw_metadata, raw_metadata_file)

    metadata = make_metadata(raw_metadata)
    print("dumping " + METADATA_FILE)
    with open(METADATA_FILE, 'w') as metadata_file:
        json.dump(metadata, metadata_file)

    asin_index_map, index_asin_map = make_target_vector_map(raw_metadata)
    print("dumping " + ASIN_INDEX_FILE)
    with open(ASIN_INDEX_FILE, 'w') as asin_index_file:
        json.dump(asin_index_map, asin_index_file)
    print("dumping " + INDEX_ASIN_FILE)
    with open(INDEX_ASIN_FILE, 'w') as index_asin_file:
        json.dump(index_asin_map, index_asin_file)

    print("Done processing metadata!")
