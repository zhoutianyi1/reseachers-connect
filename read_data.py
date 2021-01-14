import json 

objectList = []

#since the json file is too large, we will limit the num of objects
num_limit = 1000
count = 0

print("Started Reading JSON file which contains multiple JSON document")
with open('dblp_parser_python/data.json')  as f:

    for jsonObj in f:
        if num_limit == 0:
            break
        else:
            paperDict = json.loads(jsonObj)
            objectList.append(paperDict)
            count += 1
            print(count)
            # num_limit -= 1

# now we will take the title and author information
paper_info = []

for obj in objectList:
    paper_author_pair = {}
    key1 = 'author'
    if key1 in obj:
        paper_author_pair['author'] = obj['author']
    else:
        paper_author_pair['author'] = []

    key2 = 'title'
    if key2 in obj:
        paper_author_pair['title'] = obj['title']
    else:
        paper_author_pair['title'] = []

    paper_info.append(paper_author_pair)
    

f.close()