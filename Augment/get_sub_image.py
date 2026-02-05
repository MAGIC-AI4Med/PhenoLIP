import json
import os
from PIL import Image
import requests
from io import BytesIO

def download_image(url):
    """下载图像并返回PIL Image对象"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def load_local_image(path):
    """加载本地图像"""
    try:
        return Image.open(path)
    except Exception as e:
        print(f"Error loading local image from {path}: {e}")
        return None

def crop_and_save_boxes(image, detection_results, visualization, pmc_id, figure_id, base_path="/mnt/petrelfs/liangcheng/rarevisual_s2_new"):
    """根据检测结果裁剪图像并保存"""
    if not detection_results or not visualization.get('box_labels'):
        print(f"No detection results or box labels found for {pmc_id}/{figure_id}")
        return
    
    # 创建保存路径
    save_dir = os.path.join(base_path, pmc_id, figure_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取box信息
    box_labels = visualization['box_labels']
    
    for box_info in box_labels:
        box_id = f"box_{box_info['label']}"
        coordinates = box_info['coordinates']
        
        # 确保坐标值为正数，并处理边界情况
        x1 = max(0, int(coordinates[0]))
        y1 = max(0, int(coordinates[1]))
        x2 = min(image.width, int(coordinates[2]))
        y2 = min(image.height, int(coordinates[3]))
        
        # 检查坐标是否有效
        if x2 <= x1 or y2 <= y1:
            print(f"Invalid coordinates for {box_id} in {pmc_id}/{figure_id}: {coordinates}")
            continue
        
        try:
            # 裁剪图像
            cropped_image = image.crop((x1, y1, x2, y2))
            
            # 保存路径
            save_path = os.path.join(save_dir, f"{box_id}.jpg")
            
            # 保存图像
            cropped_image.save(save_path, "JPEG", quality=95)
            # print(f"Saved: {save_path}")
            
        except Exception as e:
            print(f"Error cropping/saving {box_id} for {pmc_id}/{figure_id}: {e}")

def save_full_image_by_alignments(image, alignments, pmc_id, figure_id, base_path="/mnt/petrelfs/liangcheng/rarevisual_s2_new"):
    """根据alignments将整张图像保存多份"""
    if not alignments or not alignments.get('align_list'):
        print(f"No alignments found for {pmc_id}/{figure_id}")
        return
    
    # 创建保存路径
    save_dir = os.path.join(base_path, pmc_id, figure_id)
    os.makedirs(save_dir, exist_ok=True)
    
    align_list = alignments['align_list']
    
    for align_item in align_list:
        bbox_id = align_item.get('bbox_id')
        if not bbox_id:
            continue
        
        try:
            # 保存路径
            save_path = os.path.join(save_dir, f"{bbox_id}.jpg")
            
            # 保存整张图像
            image.save(save_path, "JPEG", quality=95)
            print(f"Saved full image as: {save_path}")
            
        except Exception as e:
            print(f"Error saving full image as {bbox_id} for {pmc_id}/{figure_id}: {e}")

def process_figure(figure_info, pmc_id):
    """处理单个图像"""
    figure_id = figure_info.get('id')
    detection_results = figure_info.get('detection_results')
    visualization = figure_info.get('visualization', {})
    alignments = figure_info.get('alignments', {})
    
    if not figure_id:
        return
    
    # 尝试加载图像
    image = None
    
    # 首先尝试本地下载的图像
    downloaded_image_path = figure_info.get('downloaded_main_image')
    if downloaded_image_path and os.path.exists(downloaded_image_path):
        image = load_local_image(downloaded_image_path)
    
    # 如果本地图像不存在，尝试从URL下载
    if image is None:
        image_url = figure_info.get('image_url')
        if image_url:
            image = download_image(image_url)
    
    if image is None:
        print(f"Could not load image for {pmc_id}/{figure_id}")
        return
    
    # 检查是否有检测结果
    has_detections = (detection_results and 
                     len(detection_results) > 0 and 
                     detection_results[0].get('boxes') and 
                     len(detection_results[0]['boxes']) > 0 and
                     visualization.get('box_labels') and 
                     len(visualization['box_labels']) > 0)
    
    if has_detections:
        # 有检测结果，按检测框裁剪并保存
        # print(f"Processing with detections: {pmc_id}/{figure_id}")
        crop_and_save_boxes(image, detection_results, visualization, pmc_id, figure_id)
    else:
        # 没有检测结果，根据alignments保存整张图像
        print(f"No detections found, saving full image by alignments: {pmc_id}/{figure_id}")
        save_full_image_by_alignments(image, alignments, pmc_id, figure_id)

def process_json_data(json_file_path):
    """处理JSON数据文件"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    for item in data:
        pmc_id = item.get('pmc_id')
        if not pmc_id:
            continue
            
        crawled_info = item.get('crawled_info', [])
        
        for figure_info in crawled_info:
            process_figure(figure_info, pmc_id)

def main():
    """主函数"""
    # JSON文件路径
    json_file_path = "output/align_results_new/align_merged.json"
    
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"JSON file not found: {json_file_path}")
        return
    
    # 处理数据
    process_json_data(json_file_path)
    print("Processing completed!")

if __name__ == "__main__":
    main()