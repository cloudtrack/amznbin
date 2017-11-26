import json

from constants import METADATA_DIR, RAW_METADATA_FILE, METADATA_FILE, TOTAL_DATA_SIZE, ASIN_INDEX_FILE, INDEX_ASIN_FILE, MINIMUM_REPEAT


def make_raw_metadata():
    raw_metadata = [{}]
    for i in range(1, TOTAL_DATA_SIZE+1):
        file_path = '%s%05d.json' % (METADATA_DIR, i)
        json_data = json.loads(open(file_path).read())
        processed_json_data = {
            'TOTAL': json_data['EXPECTED_QUANTITY'],
            'DATA': {}
        }
        for bin_key in json_data['BIN_FCSKU_DATA'].keys():
            bin_meta = json_data['BIN_FCSKU_DATA'][bin_key]
            useful_meta = {
                'a': 'a',
            }
            processed_json_data['DATA'][bin_key] = useful_meta
        raw_metadata.append(processed_json_data)

    return raw_metadata


def make_metadata(raw_metadata, valid_images):
    metadata = {}
    for i in valid_images:
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
                    else:
                        metadata[asin] = {}
                        metadata[asin]['repeat'] = 1
    return metadata


def make_target_vector_map(metadata):
    asin_index_map = {}
    index = 0
    for asin in metadata.keys():
        if metadata[asin]['repeat'] >= MINIMUM_REPEAT:
            if asin not in asin_index_map.keys():
                asin_index_map[asin] = index
                index += 1
    return asin_index_map

def classify_images(asin_index_map, raw_metadata):
    valid_images = []
    for i in range(1, len(raw_metadata)):
        if raw_metadata[i]:
            bin_info = raw_metadata[i]['DATA']
            flag = True
            for bin_key in bin_info.keys():
                bin_index = asin_index_map.get(bin_key)
                if not bin_index:
                    flag = False
            if flag:
                valid_images.append(i)
    return valid_images



if __name__ == '__main__':
    mmt = TOTAL_DATA_SIZE
    rdc = TOTAL_DATA_SIZE
    raw_metadata = make_raw_metadata()
    
    valid_images = []
    for i in range(1, TOTAL_DATA_SIZE+1):
        valid_images.append(i)
    
    while mmt > 0:
        plen = len(valid_images)
        prdc = rdc
        metadata = make_metadata(raw_metadata, valid_images)
        asin_index_map = make_target_vector_map(metadata)
        valid_images = classify_images(asin_index_map, raw_metadata)
        
        rdc = plen - len(valid_images)
        mmt = mmt * 0.95 + (prdc - rdc) * 0.05
        
        print(str(len(valid_images)) + "\t" + str(len(asin_index_map)) + "\t" + str(mmt))
