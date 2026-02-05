
""" 
Modified Pipeline to detect and distinguish subfigures in a compound figure.
"""
from tqdm import tqdm
import yaml
import torch
from skimage import io
import numpy as np
import cv2
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
import os
import json
import requests  # 新增：用于下载图像
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors

from subfigure_ocr.models.yolov3 import *  # 假设这些路径与参考代码一致
from subfigure_ocr.models.network import *
from subfigure_ocr.separator import process  # 假设这些路径与参考代码一致

def visualize_subfigures(figure_path, subfigures, save_path):
    """ 可视化子图检测结果 """
    # 读取原图
    img = plt.imread(figure_path)
    
    # 创建图形和轴
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # 定义颜色列表
    color_list = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    # 为每个子图绘制边界框和标签
    for i, subfig in enumerate(subfigures):
        bbox = subfig['bbox']
        label = subfig['label']
        confidence = subfig['confidence']
        
        # 选择颜色
        color = color_list[i % len(color_list)]
        
        # 绘制边界框
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # 添加标签文本
        ax.text(x1, y1-5, f'{label} ({confidence:.3f})', 
               fontsize=12, color=color, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 设置标题
    ax.set_title(f'Detected Subfigures: {len(subfigures)} found', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")

class Classifier:
    def __init__(self):
        model_path = 'SubFig_Detection/subfigure_ocr/'  # 调整为您的路径
        configuration_file = model_path + "config/yolov3_default_subfig.cfg"
        with open(configuration_file, 'r') as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)

        self.image_size = configuration['TEST']['IMGSIZE']
        self.nms_threshold = configuration['TEST']['NMSTHRE']
        self.confidence_threshold = 0.0001
        self.dtype = torch.cuda.FloatTensor
        self.device = torch.device('cuda')

        object_detection_model = YOLOv3(configuration['MODEL'])
        self.object_detection_model = self.load_model_from_checkpoint(object_detection_model, "object_detection_model.pt")
        
        text_recognition_model = resnet152()
        self.text_recognition_model = self.load_model_from_checkpoint(text_recognition_model, 'text_recognition_model.pt')

        self.object_detection_model.eval()
        self.text_recognition_model.eval()

    def load_model_from_checkpoint(self, model, model_name):
        checkpoints_path = 'SubFig_Detection/subfigure_ocr/checkpoints/'  # 调整为您的路径
        checkpoint = checkpoints_path + model_name
        model.load_state_dict(torch.load(checkpoint))
        model.to(self.device)
        return model
    
    def detect_subfigure_boundaries(self, figure_path):
        """ Detects the bounding boxes of subfigures in figure_path """
        img = io.imread(figure_path)
        if len(np.shape(img)) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        img, info_img = process.preprocess(img, self.image_size, jitter=0)
        img = np.transpose(img / 255., (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0)
        img = Variable(img.type(self.dtype))

        img_raw = Image.open(figure_path).convert("RGB")
        width, height = img_raw.size

        with torch.no_grad():
            outputs = self.object_detection_model(img.to(self.device))
            outputs = process.postprocess(outputs, dtype=self.dtype, 
                                          conf_thre=self.confidence_threshold, nms_thre=self.nms_threshold)

        subfigure_info = []
        if outputs[0] is None:
            return subfigure_info

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs[0]:
            box = process.yolobox2label([y1.data.cpu().numpy(), x1.data.cpu().numpy(), y2.data.cpu().numpy(), x2.data.cpu().numpy()], info_img)
            box[0] = int(min(max(box[0], 0), width - 1))
            box[1] = int(min(max(box[1], 0), height - 1))
            box[2] = int(min(max(box[2], 0), width))
            box[3] = int(min(max(box[3], 0), height))
            small_box_threshold = 5
            if (box[2] - box[0] > small_box_threshold and box[3] - box[1] > small_box_threshold):
                box.append("%.3f" % (cls_conf.item()))
                subfigure_info.append(box)
        return subfigure_info

    def detect_subfigure_labels(self, figure_path, subfigure_info):
        """ Uses text recognition to read subfigure labels from figure_path """
        img_raw = Image.open(figure_path).convert("RGB")
        img_raw = img_raw.copy()
        width, height = img_raw.size

        subfigures_with_labels = []
        for subfigure in subfigure_info:
            bbox = tuple(subfigure[:4])
            img_patch = img_raw.crop(bbox)
            img_patch = np.array(img_patch)[:,:,::-1]
            img_patch, _ = process.preprocess(img_patch, 28, jitter=0)
            img_patch = np.transpose(img_patch / 255., (2, 0, 1))
            img_patch = torch.from_numpy(img_patch).type(self.dtype).unsqueeze(0)

            label_prediction = self.text_recognition_model(img_patch.to(self.device))
            label_confidence = np.amax(F.softmax(label_prediction, dim=1).data.cpu().numpy())
            x1, y1, x2, y2, box_confidence = subfigure
            total_confidence = float(box_confidence) * label_confidence
            label_value = chr(label_prediction.argmax(dim=1).data.cpu().numpy()[0] + ord("a"))
            th = 32
            if label_value == "z" :
                continue
            if (x2 - x1) < th or (y2 - y1) < th:
                print(f"Skipping label '{label_value}' due to size or invalid character.")
                # continue  # 过滤无效标签

            # 收集子图信息
            subfigures_with_labels.append({
                "label": label_value.upper(),  # e.g., 'A'
                "bbox": [x1, y1, x2, y2],
                "confidence": total_confidence,
                "geometry": [{"x": x1, "y": y1}, {"x": x1, "y": y2}, {"x": x2, "y": y1}, {"x": x2, "y": y2}]
            })
        
        # 按x坐标排序子图（从左到右，上到下）
        subfigures_with_labels.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
        return subfigures_with_labels
    
    def run(self, figure_path):
        # breakpoint()
        subfigure_info = self.detect_subfigure_boundaries(figure_path)
        subfigures_with_labels = self.detect_subfigure_labels(figure_path, subfigure_info)
        print(f"Detected Subfigures with Labels:{len(subfigures_with_labels)}") 
        print(f"Subfigures: {subfigures_with_labels}")     
        return subfigures_with_labels

def download_image(url, save_path):
    """ 下载图像如果本地不存在 """
    if not os.path.exists(save_path):
        response = requests.get(url)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded image to {save_path}")
        else:
            raise ValueError(f"Failed to download image from {url}")

def process_data(input_json_path, output_file):
    model = Classifier()
    # 提取图像信息
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for record in tqdm(data):
        if 'crawled_info' in record:
            for crawled_info in record['crawled_info']:
                if 'subfigures' in crawled_info:
                    for subfig in crawled_info['subfigures']:
                        fig_path = subfig.get('subfig_path')
                        res = process_single(model,fig_path)
                        if len(res) > 0:
                            subfig['label'] = res[0]['label']
                        else:
                            subfig['label'] = 'Not Sure'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_single(model,local_path):
    """ 处理您提供的数据JSON，区分子图 """
    
    
    
    
    # 如果本地路径不存在，下载图像
    # download_image(image_url, local_path)
    
    # 运行模型区分子图
    subfigures = model.run(local_path)
    
    # 输出结果
    print("Distinguished Subfigures:")
    for subfig in subfigures:
        subfig["confidence"] = float(subfig["confidence"])
        print(json.dumps(subfig, indent=4))
    # 可视化结果
    # if subfigures:
    #     # 创建tmp目录
    #     tmp_dir = "tmp"
    #     os.makedirs(tmp_dir, exist_ok=True)
        
    #     # 生成保存路径
    #     figure_name = os.path.splitext(os.path.basename(local_path))[0]
    #     save_path = os.path.join(tmp_dir, f"{figure_name}_subfigure_detection.jpg")
        
    #     # 可视化并保存
    #     visualize_subfigures(local_path, subfigures, save_path)
        
    #     # 同时保存JSON结果
    #     json_save_path = os.path.join(tmp_dir, f"{figure_name}_subfigures.json")
    #     with open(json_save_path, 'w', encoding='utf-8') as f:
    #         json.dump(subfigures, f, indent=4, ensure_ascii=False)
    #     print(f"JSON results saved to: {json_save_path}")
    # else:
    #     print("No subfigures detected, skipping visualization.")
    
    return subfigures

if __name__ == "__main__":
    # 您提供的数据示例
    input_file = 'output/det_output_data_with_subfigures.json'
    output_file = 'output/ocr_output.json'
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # os.makedirs(visualization_dir, exist_ok=True)
    # 处理数据并区分子图
    process_data(input_file, output_file)