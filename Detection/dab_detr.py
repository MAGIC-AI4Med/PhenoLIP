import json
import os
from tqdm import tqdm
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from PIL import Image, ImageDraw, ImageFont
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- Core Detection Logic ---

def get_model_and_processor(model_path):
    """Loads the model and processor."""
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForObjectDetection.from_pretrained(model_path)
    return processor, model

def detect_objects_in_image(image_path, processor, model):
    """
    Performs object detection on a single image.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Warning: Image file not found at {image_path}. Skipping.")
        return None
    except Exception as e:
        print(f"Warning: Could not open image {image_path}. Error: {e}. Skipping.")
        return None

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([image.size[::-1]]),
        threshold=0.5
    )
    
    # Convert results to a JSON-serializable format
    serializable_results = []
    for res in results:
        serializable_res = {
            "scores": res["scores"].tolist(),
            "labels": res["labels"].tolist(),
            "boxes": res["boxes"].tolist()
        }
        serializable_results.append(serializable_res)
        
    return serializable_results

# --- Data Processing Logic (from origin_detr.py) ---

def process_and_save_results(input_json_path, output_json_path, base_image_path, model_path):
    """
    Reads data from input_json, runs detection on images, and saves results to output_json.
    """
    print("Loading model and processor...")
    processor, model = get_model_and_processor(model_path)
    print("Model and processor loaded.")

    print(f"Reading data from {input_json_path}...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Starting detection process...")
    for record in tqdm(data, desc="Processing records"):
        if 'crawled_info' in record:
            for crawled_info in record['crawled_info']:
                image_relative_path = crawled_info.get('downloaded_main_image')
                if image_relative_path:
                    # Construct the full image path
                    image_full_path = os.path.join(base_image_path, image_relative_path)
                    
                    # Run detection
                    detection_results = detect_objects_in_image(image_full_path, processor, model)
                    
                    # Add results to the dictionary
                    if detection_results:
                        crawled_info['detection_results'] = detection_results

    print(f"Detection finished. Saving results to {output_json_path}...")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Results saved successfully.")


# --- Visualization (Optional, can be used for debugging) ---

def visualize_with_pil(image, results, save_path=None):
    """Visualizes detection results on an image using PIL."""
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    
    for res in results:
        for i, (score, label, box) in enumerate(zip(res["scores"], res["labels"], res["boxes"])):
            x1, y1, x2, y2 = box
            color = colors[i % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            label_text = f"Label {label}: {score:.2f}"
            
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - text_height - 2), label_text, fill='white', font=font)
            
    if save_path:
        img_with_boxes.save(save_path)
        print(f"Visualization saved to: {save_path}")
        
    return img_with_boxes

# --- Main Execution ---

if __name__ == "__main__":
    # Configuration
    # The JSON data path provided in the documentation
    INPUT_JSON = '/mnt/petrelfs/liangcheng/RareVisual/output/stage2_cls_results_new/filtered_cls_no_group.json'
    # Base path for the image files, assuming they are relative to the project root
    BASE_IMAGE_DIR = '/mnt/petrelfs/liangcheng/RareVisual/'
    # Path to save the output JSON with detection results
    OUTPUT_JSON = '/mnt/petrelfs/liangcheng/RareVisual/output/stage4_detection_results/detection_results.json'
    # Model path
    MODEL_PATH = '/mnt/petrelfs/liangcheng/models/pmc18m-detr'

    print("Starting object detection pipeline...")
    process_and_save_results(
        input_json_path=INPUT_JSON,
        output_json_path=OUTPUT_JSON,
        base_image_path=BASE_IMAGE_DIR,
        model_path=MODEL_PATH
    )
    print("Pipeline finished.")

    # Example of how to visualize a single result for debugging
    # To use this, you would first need to run the pipeline and then load the output JSON
    # For example:
    # with open(OUTPUT_JSON, 'r') as f:
    #     results_data = json.load(f)
    
    # # Find a record with detections to visualize
    # for record in results_data:
    #     if 'crawled_info' in record:
    #         for info in record['crawled_info']:
    #             if 'detection_results' in info and info['detection_results']:
    #                 image_path = os.path.join(BASE_IMAGE_DIR, info['downloaded_main_image'])
    #                 image = Image.open(image_path).convert("RGB")
    #                 # Note: The saved results are lists, need to be passed correctly
    #                 visualize_with_pil(image, info['detection_results'], save_path="example_detection.jpg")
    #                 # Visualize only the first valid example
    #                 exit()
