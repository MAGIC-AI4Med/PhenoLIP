import os
import json

# 你的文件列表
file_list = [
    "pubmed_downloads/res_caption_total_crawled_0_50000.json",
    # "pubmed_downloads/res_caption_total_crawled_50000_100000.json",
    "pubmed_downloads/res_caption_total_crawled_100000_150000.json",
    "pubmed_downloads/res_caption_total_crawled_150000_200000.json",
    "pubmed_downloads/res_caption_total_crawled_200000_243078.json",
]

failed_examples = []  # 存储所有失败的例子

for file_path in file_list:
    print(f"Processing: {file_path}")
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        data = json.load(f)
        # 找出失败的例子
        failed = [item for item in data if 'crawl_result' in item and 'error' in item['crawl_result']]
        print(f"  Found {len(failed)} failed examples in this file.")
        failed_examples.extend(failed)
        failed_examples.extend([item for item in data if 'crawl_result' not in item])


with open('output/res_caption_total.json','r') as f:
    data = json.load(f)
    failed_examples.extend(data[50000:100000])
# 可选：将所有失败例子保存到一个新的json文件
output_file = "output/res_caption_v1.json"
with open(output_file, 'w') as f:
    json.dump(failed_examples, f, ensure_ascii=False, indent=2)

