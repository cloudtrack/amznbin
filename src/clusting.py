import json

from constants import METADATA_DIR, RAW_METADATA_FILE, METADATA_FILE, TOTAL_DATA_SIZE, ASIN_INDEX_FILE, INDEX_ASIN_FILE, MINIMUM_REPEAT


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
        file_path = '%s%05d.json' % (METADATA_DIR, i)
        json_data = json.loads(open(file_path).read())
        processed_json_data = {
            'TOTAL': json_data['EXPECTED_QUANTITY'],
            'DATA': {},
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
    for i in range(0, len(raw_metadata)):
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
    for asin in metadata.keys():
        if asin not in asin_index_map.keys():
            asin_index_map[asin] = index
            index_asin_map[index] = asin
            index += 1
    return asin_index_map, index_asin_map

if __name__ == '__main__':
    raw_metadata = make_raw_metadata()

    metadata = make_metadata(raw_metadata)

    asin_index_map, index_asin_map = make_target_vector_map(metadata)

    image_mem = [0] * len(raw_metadata)
    metadata_mem = [0] * len(metadata)

    clusting_image_map = [[]]
    clusting_metadata_map = [[]]

    num = 1

    for i in range(0, len(index_asin_map)):
        if metadata_mem[i] == 0:
            metadata_mem[i] = 1
            iamge_list = []
            metadata_list = [index_asin_map[i]]
            image_iter = 0
            metadata_iter = 0
            while True:
                plen = len(image_list)
                
                for mi in range(metadata_iter, len(metadata_list)):
                    metadata_iter = metadata_iter + 1
                    asin = metadata_list[mi]
                    for i in metadata[asin]['bin_list']:
                        if image_mem[i] == 0:
                            image_mem[i] = 1
                            image_list.append(i)
                
                if plen == len(image_list):
                    break

                plen = len(metadata_list)

                for i in range(image_iter, len(image_list)):
                    image_iter = image_iter + 1
                    for asin in raw_metadata[i]['DATA'].keys():
                        if metadata_mem[asin_index_map[asin]] == 0:
                            metadata_mem[asin_index_map[asin]] = 1
                            metadata_list.append(asin)

                if plen == len(metadata_list):
                    break

            clusting_image_map.append(image_list)
            clusting_metadta_map.append(metadata_list)
            print(str(num) + ": " + str(len(image_list)) + " images\t" + str(len(metadata_list)) + " items")
            num = num + 1
