import json
from constants import MAXIMUM_COUNT, VALID_IMAGES_FILE, RAW_METADATA_FILE

with open(RAW_METADATA_FILE) as raw_metadata_file:
    raw_metadata = json.load(raw_metadata_file)

with open(VALID_IMAGES_FILE) as valid_images_file:
    valid_images = json.load(valid_images_file)

bin_cnt = [0]*(MAXIMUM_COUNT + 2)
for i in valid_images:
    quantity = raw_metadata[i]['TOTAL']
    if quantity > MAXIMUM_COUNT:
        bin_cnt[MAXIMUM_COUNT+1] = bin_cnt[MAXIMUM_COUNT+1] + 1
    else:
        bin_cnt[quantity] = bin_cnt[quantity] + 1

print("bin status")
for i in range(0, MAXIMUM_COUNT+1):
    print(str(i)+":\t "+str(bin_cnt[i]))
print(str(MAXIMUM_COUNT)+"up: \t "+str(bin_cnt[MAXIMUM_COUNT+1]))
