import argparse
import json
import logging
import os
from PIL import Image
import torchvision.transforms as standard_transforms
from torchvision import models
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fig_classification(fig_class_model_path):
    fig_model = models.resnext101_32x8d()
    num_features = fig_model.fc.in_features
    fc = list(fig_model.fc.children())
    fc.extend([nn.Linear(num_features, 28)])
    fig_model.fc = nn.Sequential(*fc)
    fig_model = fig_model.to(device)
    fig_model.load_state_dict(torch.load(fig_class_model_path, map_location=device))
    fig_model.eval()
    return fig_model

class RareVisionDataset(Dataset,):
    def __init__(self, data):
        self.data = []
        for item in data:
            for info in item['crawled_info']:
                if 'downloaded_main_image' in info:
                    self.data.append(info)
                else:
                    logging.warning(f"Missing 'downloaded_main_image' in info: {info}")
        # self.data = self.data[]
        mean_std = ([.485, .456, .406], [.229, .224, .225])
        self.fig_class_trasform = standard_transforms.Compose([
            standard_transforms.Resize((384, 384), interpolation=Image.Resampling.LANCZOS),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['downloaded_main_image']
        image = Image.open(img_path).convert('RGB')
        image = self.fig_class_trasform(image)
        return {'image': image, 'image_path': img_path}

def main():
    parser = argparse.ArgumentParser(description='过滤医学图片')
    parser.add_argument('--input-file', type=str, required=True, help='输入JSON文件路径')
    parser.add_argument('--start-index', type=int, default=0, help='开始索引')
    parser.add_argument('--end-index', type=int, required=True, help='结束索引')
    parser.add_argument('--output-dir', type=str, default='output', help='输出目录')
    parser.add_argument('--max-workers', type=int, default=32, help='最大线程数')
    parser.add_argument('--model-path', type=str, default='SubFig_Detection/resnext101_figure_class.pth', help='模型路径')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    args = parser.parse_args()

    # 检查输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 只处理索引区间内的数据
    with open(args.input_file, 'r') as f:
        all_data = json.load(f)
    sub_data = all_data[args.start_index:args.end_index]
    print(f"Processing records from {args.start_index} to {args.end_index} (total: {len(sub_data)})")

    # 医学类别标签（请根据实际情况调整）
    medical_labels = [15,16]  # TODO: 替换为实际医学类别索引

    dataset = RareVisionDataset(sub_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.max_workers)
    fig_model = fig_classification(args.model_path)
    predictions = {}

    for item in tqdm(dataloader):
        img_tensor = item['image'].to(device)
        image_path = item['image_path']
        fig_label = fig_model(img_tensor)
        for i in range(len(fig_label)):
            predictions[image_path[i]] = torch.argmax(fig_label[i].cpu().detach()).item()

    # 保存预测结果
    pred_file = os.path.join(
        args.output_dir,
        f"figure_classification_pred_{args.start_index}_{args.end_index}.json"
    )
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to {pred_file}")

    # 筛选原始数据
    filtered_data = []
    for article in sub_data:
        filtered_info = []
        for info in article['crawled_info']:
            img_path = info.get('downloaded_main_image', None)
            if img_path is None:
                continue
            pred_label = predictions.get(img_path, -1)
            if pred_label in medical_labels:
                filtered_info.append(info)
        if filtered_info:
            filtered_article = article.copy()
            filtered_article['crawled_info'] = filtered_info
            filtered_data.append(filtered_article)

    filtered_file = os.path.join(
        args.output_dir,
        f"filtered_medical_figures_{args.start_index}_{args.end_index}.json"
    )
    with open(filtered_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    print(f"Filtered results saved to {filtered_file}")

if __name__ == "__main__":
    main()