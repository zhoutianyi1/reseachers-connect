import json

trialData = []
count = 0
titleSet = set()
with open('data set/dblp_parser_python/dblp.json') as f:
  for i, obj in enumerate(f):
    if i > 1000000:
      break
    # convert str to object
    toAdd = {count: {}}
    try: 
      convert = eval(obj)
    except:
      continue

    if 'title' not in convert or convert['title'] == 'Home Page' or convert["title"] in titleSet: continue
    title = convert["title"]
    titleSet.add(title)
    toAdd[count]['title'] = title
    count += 1
    trialData.append(toAdd)


with open('trial_data.json', 'w') as f:
  json.dump(trialData, f, indent = 4)
