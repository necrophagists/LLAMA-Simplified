import json
json_file ="web_text_zh_testa.json"
data=[]
data_column =[]
with open(json_file,'r',encoding='utf-8') as f:
    for line in f:
       temp =json.loads(line)
       data_column.append(temp['title'])
       if temp['desc']!='' and temp['desc']!=temp['title']:
           data_column.append(temp['desc'])
       data_column.append(temp['content'])

with open('chinese_data.txt','w',encoding='utf-8') as f:
       for x in data_column:
           f.write(x)
           f.write('\n')
print("finished!")
print(len(data_column))
print(data_column[:5])