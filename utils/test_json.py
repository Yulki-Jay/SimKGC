import json
path = '/home/jiangyunqi/KGC/SimKGC/data/WN18RR/entities.json'

with open(path, 'r') as f:
    json_data = json.load(f)
    print(len(json_data))
    print(json_data[0]) # 查看一下第一个entity长什么样子
    '''
    40943
    {'entity_id': '00260881', 'entity': 'land_reform_NN_1', 
    'entity_desc': 'a redistribution of agricultural land (especially by government action)'}
    '''
    