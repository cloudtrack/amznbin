import json

valid = json.loads(open("dataset/valid_images.json").read())
invalid = json.loads(open("dataset/invalid_images.json").read())
semi = json.loads(open("dataset/semi_invalid_images.json").read())

for x in invalid :
	if x in valid :
		valid.remove(x) 

for x in semi :
	if x in valid :
		valid.remove(x) 

print("dumping valid_images.json")
with open('dataset/valid_images.json', 'w') as valid_images_file:
    json.dump(valid, valid_images_file)

print(str(len(valid)) + ' valid images left')
