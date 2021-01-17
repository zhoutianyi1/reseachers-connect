import json

trialData = []
with open('dblp_parser_python/dblp.json') as f:
  for i, obj in enumerate(f):
    if i > 10000:
      break
    # convert str to object
    toAdd = {}
    try: 
      convert = eval(obj)
    except:
      continue

    if 'author' not in convert or 'title' not in convert: continue

    toAdd["author"] = convert["author"]
    toAdd["title"] = convert["title"]
    trialData.append(toAdd)


with open('trial_data.json', 'w') as f:
  json.dump(trialData, f, indent = 4)

  