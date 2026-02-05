import os
import json
from pathlib import Path
from tqdm import tqdm
def count_valid_records(base_dir="pubmed_downloads"):
    """
    统计有效记录的数量
    
    有效记录的条件：
    1. 目录下有images子目录
    2. images目录中有jpg文件
    3. 目录下有metadata.json文件
    """
    
    if not os.path.exists(base_dir):
        print(f"错误: 目录 {base_dir} 不存在")
        return 0
    
    valid_count = 0
    total_dirs = 0
    details = []
    
    # 遍历pubmed_downloads目录下的所有子目录
    for item in tqdm(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        
        # 只处理目录
        if not os.path.isdir(item_path):
            continue
            
        total_dirs += 1
        
        # 检查条件
        has_images_dir = False
        has_jpg_files = False
        has_metadata = False
        jpg_count = 0
        
        # 1. 检查是否有images目录
        images_dir = os.path.join(item_path, "images")
        if os.path.exists(images_dir) and os.path.isdir(images_dir):
            has_images_dir = True
            
            # 2. 检查images目录中是否有jpg文件
            try:
                for file in os.listdir(images_dir):
                    if file.lower().endswith(('.jpg', '.jpeg')):
                        has_jpg_files = True
                        jpg_count += 1
            except PermissionError:
                print(f"警告: 无法访问 {images_dir}")
        
        # 3. 检查是否有metadata.json文件
        metadata_file = os.path.join(item_path, f"{item}_metadata.json")
        if os.path.exists(metadata_file):
            has_metadata = True
        
        # 判断是否为有效记录
        is_valid = has_images_dir and has_jpg_files and has_metadata
        
        if is_valid:
            valid_count += 1
        
        # 记录详细信息
        details.append({
            'directory': item,
            'has_images_dir': has_images_dir,
            'has_jpg_files': has_jpg_files,
            'jpg_count': jpg_count,
            'has_metadata': has_metadata,
            'is_valid': is_valid
        })
    
    return valid_count, total_dirs, details

def print_detailed_report(base_dir="pubmed_downloads"):
    """打印详细报告"""
    
    print(f"正在扫描目录: {base_dir}")
    print("=" * 60)
    
    valid_count, total_dirs, details = count_valid_records(base_dir)
    
    print(f"总共扫描目录数: {total_dirs}")
    print(f"有效记录数: {valid_count}")
    print(f"有效率: {valid_count/total_dirs*100:.1f}%" if total_dirs > 0 else "有效率: 0%")
    print("=" * 60)
    
    # 统计各种情况
    has_images = sum(1 for d in details if d['has_images_dir'])
    has_jpg = sum(1 for d in details if d['has_jpg_files'])
    has_metadata = sum(1 for d in details if d['has_metadata'])
    total_jpg_files = sum(d['jpg_count'] for d in details)
    
    print(f"统计详情:")
    print(f"  - 有images目录的: {has_images} ({has_images/total_dirs*100:.1f}%)")
    print(f"  - 有jpg文件的: {has_jpg} ({has_jpg/total_dirs*100:.1f}%)")
    print(f"  - 有metadata文件的: {has_metadata} ({has_metadata/total_dirs*100:.1f}%)")
    print(f"  - jpg文件总数: {total_jpg_files}")
    print("=" * 60)
    
    # 显示前10个无效记录的原因
    invalid_records = [d for d in details if not d['is_valid']]
    if invalid_records:
        print("无效记录示例 (前10个):")
        for i, record in enumerate(invalid_records[:10], 1):
            reasons = []
            if not record['has_images_dir']:
                reasons.append("缺少images目录")
            if not record['has_jpg_files']:
                reasons.append("缺少jpg文件")
            if not record['has_metadata']:
                reasons.append("缺少metadata文件")
            
            print(f"  {i}. {record['directory']}: {', '.join(reasons)}")
    
    return valid_count, total_dirs

def export_results_to_json(base_dir="pubmed_downloads", output_file="scan_results.json"):
    """将结果导出到JSON文件"""
    
    valid_count, total_dirs, details = count_valid_records(base_dir)
    
    results = {
        'scan_date': str(Path().absolute()),
        'base_directory': base_dir,
        'summary': {
            'total_directories': total_dirs,
            'valid_records': valid_count,
            'invalid_records': total_dirs - valid_count,
            'validity_rate': round(valid_count/total_dirs*100, 2) if total_dirs > 0 else 0
        },
        'details': details
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已导出到: {output_file}")
    return output_file

if __name__ == "__main__":
    # 执行扫描并打印报告
    valid_count, total_dirs = print_detailed_report()
    
    # 导出结果到JSON文件
    export_results_to_json()
    
    print(f"\n最终结果: 在 {total_dirs} 个目录中找到 {valid_count} 个有效记录")