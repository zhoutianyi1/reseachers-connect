import pandas
import json

# df = pandas.dataframe()

all_data = []
trainable_data = {} #len(author) >= 2
author_paper = {}
author_has_more_than_two_paper = []

f = open("../data set/V1.txt", "r")
block = []
list_of_block = []
for x in f:
    if x == '\n':
        list_of_block.append(block)
        block = []
    else:
        block.append(x)

def readblock():
    for block in list_of_block:
        paper_dict = {}

        paper_id = ""
        title = ""
        year = ""
        abstract = ""
        author = []
        ref = []
        for i in block:
            if type(i) != str:
                continue

            if len(i) < 2:
                continue

            length = len(i)
            i = i[:length-1]

            if i[:2] == "#*":
                title = i[2:]
            elif i[:2] == "#@":
                content = i[2:]
                author = content.split(",")
            elif i[:2] == "#t":
                year = i[2:6]
            elif len(i) > 6:
                if i[:6] == "#index":
                    paper_id = i[6:]
            elif i[:2] == "#%":
                ref.append(i[2:])
            elif i[:2] == "#!":
                abstract = i[2:]
        paper_dict = {"title":title, "paper_id":paper_id, "year":year, "abstract":abstract, "author":author, "ref":ref }
        if title!="" and len(author) >= 2:
            trainable_data[paper_dict["paper_id"]] = paper_dict
        all_data.append(paper_dict)
    return



def build_author_dict():
    for i in trainable_data:
        author_lst = trainable_data[i]["author"]
        for author in author_lst:
            if author not in author_paper:
                author_paper[author] = [trainable_data[i]["paper_id"]]
            else:
                author_paper[author].append(trainable_data[i]["paper_id"])
                author_has_more_than_two_paper.append(author)
    return

        
readblock()
build_author_dict()

# print(all_data[:10])
# print(trainable_data)
for i in range(2):
    author = author_has_more_than_two_paper[i]
    print("author", author)
    paper1 = author_paper[author][0]
    paper2 = author_paper[author][1]
    # print("paper 1", trainable_data[paper1])
    # print("paper 2", trainable_data[paper2])

print(len(author_has_more_than_two_paper))
