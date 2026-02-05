import json
import os
from PIL import Image, ImageDraw, ImageFont
import string
from tqdm import tqdm

def visualize_detection_results(json_file_path, output_base_dir="output/stage4_detection_results"):
    """
    读取检测结果JSON文件并生成可视化图像，同时将可视化路径添加到原始数据中
    
    Args:
        json_file_path: JSON文件路径
        output_base_dir: 输出基础目录
    """
    
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 用于跟踪是否有更新
    has_updates = False
    
    # 处理每个PMC条目
    for pmc_entry in data:
        pmc_id = pmc_entry.get('pmc_id', '')
        crawled_info = pmc_entry.get('crawled_info', [])
        
        # 处理每个图像
        for image_info in crawled_info:
            if 'detection_results' not in image_info or not image_info['detection_results']:
                continue
                
            # 获取图像路径
            downloaded_image = image_info.get('downloaded_main_image', '')
            if not downloaded_image or not os.path.exists(downloaded_image):
                print(f"图像文件不存在: {downloaded_image}")
                continue
            
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
                print(f"无法加载图像 {downloaded_image}: {e}")
                continue
            
            # 创建绘制对象
            draw = ImageDraw.Draw(image)
            
            # 获取检测结果
            detection_results = image_info['detection_results'][0]  # 假设只有一个检测结果
            boxes = detection_results.get('boxes', [])
            scores = detection_results.get('scores', [])
            
            # 设置字体（尝试使用系统字体，如果失败则使用默认字体）
            font_size = 34  # 增大字体
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-BoldOblique.ttf", font_size)
            
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
                # print(f"已保存可视化结果: {output_path}")
                
                # 将可视化信息添加到原始数据中
                image_info['visualization'] = {
                    'vis_image_path': output_path,
                    'vis_created': True,
                    'box_labels': box_labels,
                    'total_detections': len(boxes)
                }
                
                # 标记有更新
                has_updates = True
                
                # 打印检测信息
                # print(f"  - PMC ID: {pmc_id}")
                # print(f"  - Figure ID: {figure_id}")
                # print(f"  - 检测到 {len(boxes)} 个目标")
                # for box_info in box_labels:
                #     print(f"    Box {box_info['label']}")
                # print()
                
            except Exception as e:
                print(f"保存图像失败 {output_path}: {e}")
                # 即使保存失败也记录尝试
                image_info['visualization'] = {
                    'vis_image_path': output_path,
                    'vis_created': False,
                    'error': str(e),
                    'box_labels': box_labels,
                    'total_detections': len(boxes)
                }
                has_updates = True
    
    # 如果有更新，保存修改后的JSON文件
    if has_updates:
        # 创建备份
        backup_path = json_file_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"已创建原始文件备份: {backup_path}")
        
        # 保存更新后的文件
        with open("output/stage4_detection_results/detection_results_vis.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"已更新JSON文件: detection_results_vis.json")
        
        # 同时保存一个带时间戳的版本
        import datetime
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
    updated_data = visualize_detection_results(json_file_path)
    print("处理完成!")
    
    # 统计处理结果
    total_images = 0
    visualized_images = 0
    
    for pmc_entry in updated_data:
        for image_info in pmc_entry.get('crawled_info', []):
            if 'detection_results' in image_info and image_info['detection_results']:
                total_images += 1
                if 'visualization' in image_info and image_info['visualization'].get('vis_created', False):
                    visualized_images += 1
    
    print(f"\n处理统计:")
    print(f"总共有检测结果的图像: {total_images}")
    print(f"成功生成可视化的图像: {visualized_images}")
    print(f"成功率: {visualized_images/total_images*100:.1f}%" if total_images > 0 else "成功率: 0%")

if __name__ == "__main__":
    main()