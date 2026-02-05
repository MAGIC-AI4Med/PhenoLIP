import json
import os
import re
from pathlib import Path
from tqdm import tqdm
def extract_pmc_id_from_path(file_path):
    """从文件路径中提取PMC ID"""
    match = re.search(r'PMC(\d+).json', file_path)
    if match:
        return match.group(1)
    return None

def extract_pmc_id_from_dirname(dirname):
    """从目录名中提取PMC ID"""
    match = re.search(r'pmc_(\d+)', dirname)
    if match:
        return match.group(1)
    return None

def consolidate_crawled_data():
    """整理所有爬取的结果"""
    
    # 1. 读取原始的疾病匹配文件
    caption_file = "output/res_caption_total.json"
    if not os.path.exists(caption_file):
        print(f"Error: {caption_file} not found!")
        return None
    
    with open(caption_file, 'r', encoding='utf-8') as f:
        disease_data = json.load(f)
    
    # 2. 创建PMC ID到疾病信息的映射
    pmc_to_disease = {}
    for item in disease_data:
        if item.get('is_disease_related', False):
            pmc_id = extract_pmc_id_from_path(item['file_name'])
            if pmc_id:
                pmc_to_disease[pmc_id] = {
                    'abb_matches': item.get('abb_matches', []),
                    'full_matches': item.get('full_matches', []),
                    'total_matches': item.get('total_matches', 0)
                }
    
    # 3. 遍历pubmed_downloads目录
    downloads_dir = "pubmed_downloads"
    if not os.path.exists(downloads_dir):
        print(f"Error: {downloads_dir} directory not found!")
        return None
    
    consolidated_results = []
    
    for subdir in tqdm(os.listdir(downloads_dir)):
        subdir_path = os.path.join(downloads_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        # 提取PMC ID
        pmc_id = extract_pmc_id_from_dirname(subdir)
        if not pmc_id:
            print(f"Warning: Could not extract PMC ID from directory: {subdir}")
            continue
        
        # 检查是否在疾病相关数据中
        if pmc_id not in pmc_to_disease:
            print(f"Warning: PMC ID {pmc_id} not found in disease data")
            continue
        
        # 读取metadata文件
        metadata_file = os.path.join(subdir_path, f"{subdir}_metadata.json")
        if not os.path.exists(metadata_file):
            print(f"Warning: Metadata file not found for {subdir}")
            continue
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                crawled_info = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading metadata for {subdir}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error reading {metadata_file}: {e}")
            continue
        
        # 验证图片文件是否存在
        images_dir = os.path.join(subdir_path, "images")
        for item in crawled_info:
            if 'downloaded_main_image' in item:
                image_path = item['downloaded_main_image']
                if not os.path.exists(image_path):
                    print(f"Warning: Image file not found: {image_path}")
        
        # 构建结果记录
        result_record = {
            'pmc_id': pmc_id,
            'abb_matches': pmc_to_disease[pmc_id]['abb_matches'],
            'full_matches': pmc_to_disease[pmc_id]['full_matches'],
            'total_matches': pmc_to_disease[pmc_id]['total_matches'],
            'crawled_info': crawled_info
        }
        
        consolidated_results.append(result_record)
    
    return consolidated_results

def save_consolidated_results(results, output_file="consolidated_crawled_data.json"):
    """保存整理后的结果"""
    if results is None:
        print("No results to save")
        return
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(results)} records to {output_file}")
        
        # 打印统计信息
        total_images = sum(len(record['crawled_info']) for record in results)
        print(f"Total papers processed: {len(results)}")
        print(f"Total images: {total_images}")
        
        # 统计疾病匹配信息
        total_abb_matches = sum(len(record['abb_matches']) for record in results)
        total_full_matches = sum(len(record['full_matches']) for record in results)
        print(f"Total abbreviation matches: {total_abb_matches}")
        print(f"Total full name matches: {total_full_matches}")
        
    except Exception as e:
        print(f"Error saving results: {e}")

def validate_data_integrity(results):
    """验证数据完整性"""
    print("\n=== Data Integrity Check ===")
    
    issues = []
    
    for i, record in enumerate(results):
        pmc_id = record['pmc_id']
        
        # 检查必要字段
        required_fields = ['pmc_id', 'abb_matches', 'full_matches', 'crawled_info']
        for field in required_fields:
            if field not in record:
                issues.append(f"Record {i} (PMC {pmc_id}): Missing field '{field}'")
        
        # 检查爬取信息
        for j, info in enumerate(record.get('crawled_info', [])):
            if 'downloaded_main_image' in info:
                image_path = info['downloaded_main_image']
                if not os.path.exists(image_path):
                    issues.append(f"Record {i} (PMC {pmc_id}), Image {j}: File not found - {image_path}")
            
            # 检查必要的图片信息字段
            required_image_fields = ['id', 'caption', 'image_url']
            for field in required_image_fields:
                if field not in info:
                    issues.append(f"Record {i} (PMC {pmc_id}), Image {j}: Missing field '{field}'")
    
    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues[:10]:  # 只显示前10个问题
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
    else:
        print("No data integrity issues found!")
    
    return len(issues) == 0

def main():
    """主函数"""
    print("Starting data consolidation...")
    
    # 整理数据
    breakpoint()
    results = consolidate_crawled_data()
    
    if results is None:
        print("Failed to consolidate data")
        return
    
    # 验证数据完整性
    is_valid = validate_data_integrity(results)
    
    # 保存结果
    save_consolidated_results(results)
    
    # 显示样例记录
    if results:
        print("\n=== Sample Record ===")
        sample = results[0]
        print(f"PMC ID: {sample['pmc_id']}")
        print(f"Full matches: {sample['full_matches']}")
        print(f"Number of images: {len(sample['crawled_info'])}")
        if sample['crawled_info']:
            first_image = sample['crawled_info'][0]
            print(f"First image ID: {first_image.get('id', 'N/A')}")
            print(f"Caption preview: {first_image.get('caption', 'N/A')[:100]}...")

if __name__ == "__main__":
    main()