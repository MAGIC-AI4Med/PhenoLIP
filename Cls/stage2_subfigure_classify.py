"""
Pipeline to filter out nonmedical subfigs
"""
import csv
import json
import logging
import os
from PIL import Image
import torchvision.transforms as standard_transforms
from torchvision import models
import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
import datetime
from spacy.tokens import Span
import numpy as np
import json
from torch.utils.data import Dataset

def fig_classification(fig_class_model_path):
    fig_model = models.resnext101_32x8d()
    num_features = fig_model.fc.in_features
    fc = list(fig_model.fc.children())  # Remove last layer
    fc.extend([nn.Linear(num_features, 28)])  # Add our layer with 28 outputs
    fig_model.fc = nn.Sequential(*fc)
    fig_model = fig_model.to(device)
    fig_model.load_state_dict(torch.load(fig_class_model_path))
    fig_model.eval()
    
    return fig_model

class RareVisionDataset(Dataset):
    def __init__(self, input_file):
        # 加载数据集
        with open(input_file, 'r') as f:
            data = json.load(f)
        self.data = []
        for item in data:
            for info in item['crawled_info']:
                if 'downloaded_main_image' in info:
                    self.data.append(info)
                else:
                    logging.warning(f"Missing 'downloaded_main_image' in info: {info}")
        self.data = self.data
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
        return {
            'image': image,
            'image_path': img_path
        }

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='DocFigure trained model')
    parser.add_argument('--input-file', default='data/consolidated_crawled_filtered.json', type=str)
    parser.add_argument('-bs', default=4, type=int)
    args = parser.parse_args()

    medical_labels = [15]  # Placeholder: Adjust this list to match your model's medical classes!

    # Load the original data for filtering later
    with open(args.input_file, 'r') as f:
        original_data = json.load(f)
    # 打印使用的gpu号
    print(f"Using device: { torch.cuda.current_device()}")
    # Calculate original statistics
    original_articles = len(original_data)
    original_images = sum(len(article['crawled_info']) for article in original_data)
    print(f"Original: {original_articles} articles, {original_images} images")

    # Load the dataset
    dataset = RareVisionDataset(args.input_file)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=32)
    
    # Load the model
    fig_model = fig_classification('SubFig_Detection/resnext101_figure_class.pth')
    predictions = {}
    for item in tqdm(dataloader):
        img_tensor = item['image'].to(device)
        image_path = item['image_path']
        
        fig_label = fig_model(img_tensor)
        for i in range(len(fig_label)):
            predictions[image_path[i]] = torch.argmax(fig_label[i].cpu().detach()).item()
    # 保存预测结果
    with open('output/tmp_figure_classification_prediction_results.json', 'w') as f:
        json.dump(predictions, f, indent=4)
    # Now filter the original data based on predictions
    filtered_data = []
    for article in original_data:
        filtered_info = []
        for info in article['crawled_info']:
            if 'downloaded_main_image' not in info:
                logging.warning(f"Missing 'downloaded_main_image' in info: {info}")
                continue
            img_path = info['downloaded_main_image']
            pred_label = predictions.get(img_path, -1)  # Default to -1 if not found (though it should be present)
            if pred_label in medical_labels:
                filtered_info.append(info)
        
        if filtered_info:  # Only keep the article if it has at least one medical image
            filtered_article = article.copy()  # Preserve original structure
            filtered_article['crawled_info'] = filtered_info
            filtered_data.append(filtered_article)

    # Calculate filtered statistics
    filtered_articles = len(filtered_data)
    filtered_images = sum(len(article['crawled_info']) for article in filtered_data)
    print(f"Filtered: {filtered_articles} articles, {filtered_images} images")

    # Save the filtered data
    output_file = 'output/filtered_medical_figures.json'  # Adjust output file name as needed
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    print(f"Filtered results saved to {output_file}")

    # Optionally save predictions for reference
    pred_output_file = 'output/figure_classification_results.json'
    with open(pred_output_file, 'w') as f:
        json.dump(predictions, f, indent=4)
    print(f"Figure classification results saved to {pred_output_file}")