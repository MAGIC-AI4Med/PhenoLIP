"""
统计全文关键词匹配之后的文章数量"""
import os
from tqdm import tqdm
import json
input_directory='output/filter_caption_with_disease'
# 收集所有JSON文件
json_files = []
match_data = {}
# mapping 
id2name_file = 'data/Orpha/orpha2name.json'
with open(id2name_file, 'r', encoding='utf-8') as f:
    id2name = json.load(f)
for root, dirs, files in os.walk(input_directory):
    for file in tqdm(files):
        if file.endswith('.json'):
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                full_data= json.load(f)
                full_data = full_data['filtered_results']
                for data in full_data:
                    json_files.append(data['file_name'])
                    abb_matches = data.get('abb_matches', [])
                    full_matches = data.get('full_matches', [])
                    total_matches = abb_matches + full_matches
                    # 对total_matches进行去重（将list转为tuple）
                    total_matches = list(set(tuple(x) for x in total_matches))
                    # 将Match的信息添加到match_data中
                    for disease_name, disease_id in total_matches:
                        match_data[disease_id] = match_data.get(disease_id, 0) + 1
                
# 对match_data进行排序
match_data = dict(sorted(match_data.items(), key=lambda item: item[1], reverse=True))
# 将疾病ID转换为名称
match_name_data = {id2name.get(disease_id, disease_id): count for disease_id, count in match_data.items()}
#保存文件列表
# with open('data/filtered_with_full_list.json', 'w') as f:
#     json.dump(json_files, f, indent=4)
# # 保存匹配数据
# with open('data/match_data_full.json', 'w') as f:
#     json.dump(match_data, f, indent=4)
# # 保存匹配名称数据
# with open('data/match_name_data_full.json', 'w') as f:
#     json.dump(match_name_data, f, indent=4)
with open('data/filtered_with_caption_list.json', 'w') as f:
    json.dump(json_files, f, indent=4)
# 保存匹配数据
with open('data/match_data_caption.json', 'w') as f:
    json.dump(match_data, f, indent=4)
# 保存匹配名称数据
with open('data/match_name_data_caption.json', 'w') as f:
    json.dump(match_name_data, f, indent=4)