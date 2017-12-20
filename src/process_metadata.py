import json

from constants import METADATA_DIR, RAW_METADATA_FILE, METADATA_FILE, TOTAL_DATA_SIZE, ASIN_INDEX_FILE, INDEX_ASIN_FILE, VALID_IMAGES_FILE, MINIMUM_REPEAT, MAXIMUM_IMAGE_NUM, MAXIMUM_COUNT, BALANCE_RATE


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


def make_metadata(raw_metadata, valid_images):
    metadata = {}
#    cnt = 0
    for i in valid_images:
#        if cnt % 1000 == 0:
#            print("make_metadata - processing (%d/%d)..." % (cnt, len(valid_images)))
#        cnt = cnt + 1
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

def make_target_vector_map(metadata, isClustering):
    asin_index_map = {}
    index_asin_map = {}
    index = 1    # 0 is for empty bin
#    i = 0
    for asin in metadata.keys():
#        if i % 1000 == 0:
#            print("make_target_vector_map - processing (%d/%d)..." % (i, len(metadata.keys())))
#        i += 1
        if isClustering or (metadata[asin]['repeat'] >= MINIMUM_REPEAT):
            if asin not in asin_index_map.keys():
                asin_index_map[asin] = index
                index_asin_map[index] = asin
                index += 1
    return asin_index_map, index_asin_map

def classify_images(asin_index_map, raw_metadata, pvalid_images):
    valid_images = []
    for i in pvalid_images:
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

def count_bin(raw_metadata, valid_images):
    bin_cnt = [0]*(MAXIMUM_COUNT + 2)
    for i in valid_images:
        quantity = raw_metadata[i]['TOTAL']
        if quantity > MAXIMUM_COUNT:
            bin_cnt[MAXIMUM_COUNT+1] = bin_cnt[MAXIMUM_COUNT+1] + 1
        else:
            bin_cnt[quantity] = bin_cnt[quantity] + 1
    return bin_cnt


if __name__ == '__main__':
    raw_metadata = make_raw_metadata()
    print("dumping " + RAW_METADATA_FILE)
    with open(RAW_METADATA_FILE, 'w') as raw_metadata_file:
        json.dump(raw_metadata, raw_metadata_file)

    valid_images = range(1, len(raw_metadata))

    metadata = make_metadata(raw_metadata, valid_images)

    asin_index_map, index_asin_map = make_target_vector_map(metadata, True)

    image_mem = [0] * len(raw_metadata)
    metadata_mem = [0] * len(metadata)

    clustering_image_list = []
    bin_cnt = []

    for i in range(0, len(index_asin_map)):
        if metadata_mem[i] == 0:
            metadata_mem[i] = 1
            image_list = []
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

                for ii in range(image_iter, len(image_list)):
                    image_iter = image_iter + 1
                    i = image_list[ii]
                    for asin in raw_metadata[i]['DATA'].keys():
                        if metadata_mem[asin_index_map[asin]] == 0:
                            metadata_mem[asin_index_map[asin]] = 1
                            metadata_list.append(asin)

                if plen == len(metadata_list):
                    break
            if len(image_list) > len(clustering_image_list):
                clustering_image_list = image_list

    valid_images = clustering_image_list
    for i in range(1, len(raw_metadata)):
        quantity = raw_metadata[i]['TOTAL']
        if quantity == 0:
            valid_images.append(i)
    it = 0

    while len(valid_images) > MAXIMUM_IMAGE_NUM:
        it = it + 1
        
        print("make metadata, target vector map, valid images (iteration "+str(it)+")")
        
        metadata = make_metadata(raw_metadata, valid_images)

        asin_index_map, index_asin_map = make_target_vector_map(metadata, False)

        valid_images = classify_images(asin_index_map, raw_metadata, valid_images)


        print("data balancing")
        index = 0
        bin_cnt = count_bin(raw_metadata, valid_images)

        bb = [0]*(MAXIMUM_COUNT+2)
        while True:
            if index >= len(valid_images):
                index = 0
            bbt = False
            for i in range(0, MAXIMUM_COUNT+2):
                bb[i] = (bin_cnt[i] > len(valid_images)*BALANCE_RATE)
                bbt = (bbt or bb[i])
            if not bbt:
                break
            image = valid_images[index]
            quantity = raw_metadata[image]['TOTAL']
            if quantity > MAXIMUM_COUNT:
                if bb[MAXIMUM_COUNT+1]:
                    del valid_images[index]
                    bin_cnt[MAXIMUM_COUNT+1] = bin_cnt[MAXIMUM_COUNT+1] - 1
                    index = index - 1
            else:
                if bb[quantity]:
                    del valid_images[index]
                    bin_cnt[quantity] = bin_cnt[quantity] - 1
                    index = index - 1
            index = index + 1

        print("valid images: "+str(len(valid_images))+"\tobjects: "+str(len(asin_index_map)))
        print("bin status")
        bin_cnt = count_bin(raw_metadata, valid_images)
        for i in range(0, MAXIMUM_COUNT+1):
            print(str(i)+":\t "+str(bin_cnt[i]))
        print(str(MAXIMUM_COUNT)+"up: \t "+str(bin_cnt[MAXIMUM_COUNT+1]))

    print("dumping " + METADATA_FILE)
    with open(METADATA_FILE, 'w') as metadata_file:
        json.dump(metadata, metadata_file)

    print("dumping " + ASIN_INDEX_FILE)
    with open(ASIN_INDEX_FILE, 'w') as asin_index_file:
        json.dump(asin_index_map, asin_index_file)
    print("dumping " + INDEX_ASIN_FILE)
    with open(INDEX_ASIN_FILE, 'w') as index_asin_file:
        json.dump(index_asin_map, index_asin_file)

    print("dumping " + VALID_IMAGES_FILE)
    with open(VALID_IMAGES_FILE, 'w') as valid_images_file:
        json.dump(valid_images, valid_images_file)

    print("Done processing metadata!")
    print("images: " + str(len(valid_images)))
    print("objects: " + str(len(asin_index_map)))
    bin_cnt = count_bin(raw_metadata, valid_images)

    print("bin status")
    for i in range(0, MAXIMUM_COUNT+1):
        print(str(i)+":\t "+str(bin_cnt[i]))
    print(str(MAXIMUM_COUNT)+"up: \t "+str(bin_cnt[MAXIMUM_COUNT+1]))

