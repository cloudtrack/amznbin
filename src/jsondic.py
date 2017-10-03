import json

from constants import ASIN_INDEX_FILE, INDEX_ASIN_FILE, RAW_METADATA_FILE, METADATA_FILE


def json2tv(index_list):
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
        tv_list.append(tv)
    return tv_list


def tv2res(tv):
    with open(METADATA_FILE) as metadata_file:
        metadata = json.load(metadata_file)
    with open(INDEX_ASIN_FILE) as index_asin_file:
        index_asin_map = json.load(index_asin_file)
    res = {}
    for i in range(len(tv)):
        if tv[i] != 0:
            asin = index_asin_map[str(i)]
            asin_meta = {
                'name': metadata[asin]['name'],
                'quantity': tv[i],
            }
            res[asin] = asin_meta
    return res
