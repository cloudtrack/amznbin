import json

RAW_METADATA_FILE = "dataset/raw_metadata.json"
METADATA_FILE = "dataset/metadata.json"
ASIN_INDEX_FILE = "dataset/asin_index_map.json"
INDEX_ASIN_FILE = "dataset/index_asin_map.json"


# make dic and save to json
def make_dic():
    print("load raw_metadata.json file")
    with open(RAW_METADATA_FILE) as raw_metadata_file:
        raw_metadata = json.load(raw_metadata_file)
    total_count = len(raw_metadata)
    asin_index_map = {}
    index_asin_map = {}
    index = 0
    count = 0
    for data in raw_metadata:
        if count % 1000 == 0:
            print("make asin:index map, processing (%d/%d)..." % (count, total_count))
        for asin in data['DATA'].keys():
            if asin not in asin_index_map.keys():
                asin_index_map[asin] = index
                index_asin_map[index] = asin
                index += 1
        count += 1

    print("dumping asin_index_file")
    asin_index_file = open(ASIN_INDEX_FILE, 'w')
    json.dump(asin_index_map, asin_index_file)
    asin_index_file.close()

    print("dumping index_asin_file")
    index_asin_file = open(INDEX_ASIN_FILE, 'w')
    json.dump(index_asin_map, index_asin_file)
    index_asin_file.close()
    print("Done processing target vector data")

if __name__ == '__main__':
    make_dic()


def get_tv_list(index_list):
    with open(RAW_METADATA_FILE) as raw_metadata_file:
        raw_metadata = json.load(raw_metadata_file)
    with open(ASIN_INDEX_FILE) as asin_index_file:
        asin_index_map = json.load(asin_index_file)
    tv_list = []
    for index in index_list:
        tv = [0] * len(asin_index_map.keys())
        data = raw_metadata[index]
        for asin in data['DATA'].keys():
            tv_index = asin_index_map.get(asin)
            tv[tv_index] = data['DATA'][asin]['quantity']
        tv_list.append(tv_list)
    return tv_list


def tv2res(tv):
    with open(METADATA_FILE) as metadata_file:
        metadata = json.load(metadata_file)
    with open(INDEX_ASIN_FILE) as index_asin_file:
        index_asin_map = json.load(index_asin_file)
    res = {}
    for i in range(len(tv)):
        if tv[i] != 0:
            asin = index_asin_map[i]
            asin_meta = {
                'name': metadata[asin]['name'],
                'quantity': tv[i],
            }
            res[asin] = asin_meta
    return res
