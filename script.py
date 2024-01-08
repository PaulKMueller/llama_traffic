import json

with open("output/embedding_test.json") as file:
    content = json.load(file)

false_categorizations = 0
correct_categorizations = 0

false = []
correct = []

for bucket in content.keys():
    print(f"Bucket: {bucket}")
    print(content[bucket].keys())
    for key in content[bucket].keys():
        similarities = content[bucket][key]
        max_bucket = max(zip(similarities.values(), similarities.keys()))[1]
        print(similarities)
        print(max_bucket)
        if max_bucket == bucket:
            correct_categorizations += 1
            correct.append(bucket)
        else:
            false_categorizations += 1
            false.append(bucket)

print(correct_categorizations)
d = {x: correct.count(x) for x in correct}
print(d)
# print(correct)
print(false_categorizations)
d = {x: false.count(x) for x in false}
print(d)
# print(false)
