import json
import os
from PIL import Image, ImageDraw, ImageFont
import string
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from typing import Tuple, Dict, Any
import datetime

def process_single_image(task_data: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    处理单个图像的检测结果可视化
    
    Args:
        task_data: 包含图像信息和相关参数的字典
        
    Returns:
        Tuple[bool, str, Dict[str, Any]]: (是否成功, 消息, 更新后的图像信息)
    """
    try:
        image_info = task_data['image_info']
        pmc_id = task_data['pmc_id']
        
        if 'detection_results' not in image_info or not image_info['detection_results']:
            return False, f"图像 {image_info.get('id', 'unknown')} 没有检测结果", image_info
            
        # 获取图像路径
        downloaded_image = image_info.get('downloaded_main_image', '')
        if not downloaded_image or not os.path.exists(downloaded_image):
            return False, f"图像文件不存在: {downloaded_image}", image_info
        
        # 创建vis目录
        image_dir = os.path.dirname(downloaded_image)
        vis_dir = os.path.join(image_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 加载原始图像
        try:
            image = Image.open(downloaded_image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            return False, f"无法加载图像 {downloaded_image}: {e}", image_info
        
        # 创建绘制对象
        draw = ImageDraw.Draw(image)
        
        # 获取检测结果
        detection_results = image_info['detection_results'][0]  # 假设只有一个检测结果
        boxes = detection_results.get('boxes', [])
        scores = detection_results.get('scores', [])
        
        # 设置字体（尝试使用系统字体，如果失败则使用默认字体）
        font_size = 34  # 增大字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-BoldOblique.ttf", font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # 存储box标签映射信息
        box_labels = []
        
        # 绘制每个检测框
        for i, (box, score) in enumerate(zip(boxes, scores)):
            # 获取框的坐标
            x1, y1, x2, y2 = box
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, image.width))
            y1 = max(0, min(y1, image.height))
            x2 = max(0, min(x2, image.width))
            y2 = max(0, min(y2, image.height))
            
            # 计算框的中心点
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 生成标签（A, B, C, ...）
            label = string.ascii_uppercase[i % 26]
            if i >= 26:
                label = string.ascii_uppercase[(i // 26) - 1] + string.ascii_uppercase[i % 26]
            label = f"box_{label}"
            # 存储box信息
            box_labels.append({
                'label': label,
                'box_index': i,
                'score': score,
                'coordinates': [x1, y1, x2, y2]
            })
            
            # 设置颜色（使用不同颜色区分不同的框）
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
            color = colors[i % len(colors)]
            
            # 绘制检测框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=6)
            
            # 计算圆圈半径 - 增大圆圈
            circle_radius = min(50, max(30, min(abs(x2-x1), abs(y2-y1)) // 6))
            
            # 绘制中心圆圈
            circle_bbox = [
                center_x - circle_radius,
                center_y - circle_radius,
                center_x + circle_radius,
                center_y + circle_radius
            ]
            draw.ellipse(circle_bbox, fill=color, outline='white', width=3)
            
            # 在圆圈中心绘制标签
            if font:
                try:
                    # 获取文本边界框来计算居中位置
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    text_x = center_x - text_width / 2
                    text_y = center_y - text_height / 2
                    
                    # 绘制白色文字，添加黑色边框效果以增强可读性
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                draw.text((text_x + dx, text_y + dy), label, fill='black', font=font)
                    draw.text((text_x, text_y), label, fill='white', font=font)
                    
                except:
                    # 如果textbbox不可用，使用估算方法
                    fallback_x = center_x - font_size // 3
                    fallback_y = center_y - font_size // 2
                    # 添加黑色边框效果
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                draw.text((fallback_x + dx, fallback_y + dy), label, fill='black', font=font)
                    draw.text((fallback_x, fallback_y), label, fill='white', font=font)
        
        # 保存可视化结果
        figure_id = image_info.get('id', 'unknown')
        output_filename = f"{pmc_id}_{figure_id}_detection_vis.jpg"
        output_path = os.path.join(vis_dir, output_filename)
        
        try:
            image.save(output_path, 'JPEG', quality=95)
            
            # 创建更新后的图像信息副本
            updated_image_info = image_info.copy()
            updated_image_info['visualization'] = {
                'vis_image_path': output_path,
                'vis_created': True,
                'box_labels': box_labels,
                'total_detections': len(boxes)
            }
            
            return True, f"成功处理图像 {pmc_id}_{figure_id} (检测到 {len(boxes)} 个目标)", updated_image_info
            
        except Exception as e:
            # 即使保存失败也记录尝试
            updated_image_info = image_info.copy()
            updated_image_info['visualization'] = {
                'vis_image_path': output_path,
                'vis_created': False,
                'error': str(e),
                'box_labels': box_labels,
                'total_detections': len(boxes)
            }
            return False, f"保存图像失败 {output_path}: {e}", updated_image_info
            
    except Exception as e:
        return False, f"处理图像时发生错误: {str(e)}", task_data.get('image_info', {})

def visualize_detection_results_multithreaded(json_file_path, output_base_dir="output/stage4_detection_results", max_workers=None):
    """
    使用多线程读取检测结果JSON文件并生成可视化图像
    
    Args:
        json_file_path: JSON文件路径
        output_base_dir: 输出基础目录
        max_workers: 最大线程数，默认为 CPU 核心数
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 准备任务列表
    tasks = []
    pmc_to_image_indices = {}  # 用于追踪每个图像在原始数据中的位置
    
    for pmc_idx, pmc_entry in enumerate(data):
        pmc_id = pmc_entry.get('pmc_id', '')
        crawled_info = pmc_entry.get('crawled_info', [])
        
        for img_idx, image_info in enumerate(crawled_info):
            if 'detection_results' in image_info and image_info['detection_results']:
                task_data = {
                    'image_info': image_info,
                    'pmc_id': pmc_id
                }
                tasks.append(task_data)
                # 记录任务索引与原始数据位置的映射
                pmc_to_image_indices[len(tasks) - 1] = (pmc_idx, img_idx)
    
    if not tasks:
        print("没有找到需要处理的图像")
        return data
    
    # 设置最大线程数
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(tasks))
    
    print(f"准备使用 {max_workers} 个线程处理 {len(tasks)} 个图像...")
    
    total_success = 0
    has_updates = False
    
    # 使用多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用 tqdm 显示进度条
        results = list(tqdm(
            executor.map(process_single_image, tasks),
            total=len(tasks),
            desc="处理图像可视化"
        ))
    
    # 处理结果并更新原始数据
    for task_idx, (success, message, updated_image_info) in enumerate(results):
        if success:
            total_success += 1
            has_updates = True
            
            # 更新原始数据中的图像信息
            pmc_idx, img_idx = pmc_to_image_indices[task_idx]
            data[pmc_idx]['crawled_info'][img_idx] = updated_image_info
        
        # 只显示失败的消息，成功的太多会刷屏
        if not success:
            print(f"失败: {message}")
    
    print(f"\n处理完成!")
    print(f"总任务数: {len(tasks)}")
    print(f"成功: {total_success}")
    print(f"失败: {len(tasks) - total_success}")
    print(f"成功率: {total_success/len(tasks)*100:.1f}%")
    
    # 如果有更新，保存修改后的JSON文件
    if has_updates:
        # 确保输出目录存在
        os.makedirs(output_base_dir, exist_ok=True)
        
        # 创建备份
        backup_path = json_file_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"已创建原始文件备份: {backup_path}")
        
        # 保存更新后的文件
        output_path = os.path.join(output_base_dir, "detection_results_vis.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"已更新JSON文件: {output_path}")
        
        # 同时保存一个带时间戳的版本
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_path = json_file_path.replace('.json', f'_with_vis_{timestamp}.json')
        with open(timestamped_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"已保存带时间戳的版本: {timestamped_path}")
    
    return data

def main():
    """主函数"""
    json_file_path = "output/stage4_detection_results/detection_results.json"
    
    if not os.path.exists(json_file_path):
        print(f"JSON文件不存在: {json_file_path}")
        return
    
    print("开始处理目标检测可视化...")
    
    # 使用多线程处理，可以自定义线程数
    updated_data = visualize_detection_results_multithreaded(
        json_file_path, 
        max_workers=8  # 可以根据需要调整线程数
    )
    
    print("全部处理完成!")
    
    # 统计最终结果
    total_images = 0
    visualized_images = 0
    
    for pmc_entry in updated_data:
        for image_info in pmc_entry.get('crawled_info', []):
            if 'detection_results' in image_info and image_info['detection_results']:
                total_images += 1
                if 'visualization' in image_info and image_info['visualization'].get('vis_created', False):
                    visualized_images += 1
    
    print(f"\n最终统计:")
    print(f"总共有检测结果的图像: {total_images}")
    print(f"成功生成可视化的图像: {visualized_images}")
    print(f"成功率: {visualized_images/total_images*100:.1f}%" if total_images > 0 else "成功率: 0%")

if __name__ == "__main__":
    main()