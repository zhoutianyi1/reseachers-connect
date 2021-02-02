import json 

objectList = []

#since the json file is too large, we will limit the num of objects for testing
num_limit = 10000

print("Started Reading JSON file which contains multiple JSON document")
with open('../data set/dblp_parser_python/dblp.json')  as f:

    for jsonObj in f:
        if num_limit == 0:
            break
        else:
            paperDict = json.loads(jsonObj)
            # print(paperDict)
            objectList.append(paperDict)
            num_limit -= 1

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
    
print(len(objectList))
f.close()

