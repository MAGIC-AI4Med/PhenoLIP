import os
from tqdm import tqdm
input_directory='/mnt/petrelfs/liangcheng/RareVisual/data/pmc_json'
# 收集所有XML文件
xml_files = []
for root, dirs, files in os.walk(input_directory):
    for file in tqdm(files):
        if file.endswith('.json'):
            xml_files.append(os.path.join(root, file))
print(len(xml_files))

# 去重
xml_files = list(set(xml_files))
print(f"去重后文件数量: {len(xml_files)}")

# 排序
xml_files.sort()
# 保存文件列表
import json
with open('data/pmc_paper_list.json', 'w') as f:
    json.dump(xml_files, f, indent=4)