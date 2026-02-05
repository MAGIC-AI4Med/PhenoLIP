import json
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

def visualize_from_json(results_json_path, base_image_path, output_dir):
    """
    Loads detection results from a JSON file, visualizes them, and saves the images.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from {results_json_path}...")
    with open(results_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("Starting visualization process...")
    for record in tqdm(data, desc="Visualizing records"):
        if 'crawled_info' in record:
            for info in record['crawled_info']:
                if 'detection_results' in info and info['detection_results']:
                    image_relative_path = info.get('downloaded_main_image')
                    if image_relative_path:
                        image_full_path = os.path.join(base_image_path, image_relative_path)
                        
                        try:
                            image = Image.open(image_full_path).convert("RGB")
                        except FileNotFoundError:
                            print(f"Warning: Image file not found at {image_full_path}. Skipping.")
                            continue
                        except Exception as e:
                            print(f"Warning: Could not open image {image_full_path}. Error: {e}. Skipping.")
                            continue

                        # Create a unique save path for the visualization
                        image_filename = os.path.basename(image_relative_path)
                        pmc_id = record.get("pmc_id", "unknown_pmc")
                        figure_id = info.get("id", "unknown_figure")
                        save_filename = f"{pmc_id}_{figure_id}_{image_filename}"
                        save_path = os.path.join(output_dir, save_filename)

                        # Visualize and save the image
                        visualize_with_pil(image, info['detection_results'], save_path=save_path)

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
        # print(f"Visualization saved to: {save_path}") # Optional: uncomment for verbose output
        
    return img_with_boxes

if __name__ == "__main__":
    # Configuration
    # The JSON data path from the detection script
    INPUT_JSON = '/mnt/petrelfs/liangcheng/RareVisual/output/stage4_detection_results/detection_results.json'
    # Base path for the image files
    BASE_IMAGE_DIR = '/mnt/petrelfs/liangcheng/RareVisual/'
    # Directory to save the visualized images
    OUTPUT_VISUALIZATION_DIR = '/mnt/petrelfs/liangcheng/RareVisual/output/stage4_visualizations/'

    print("Starting visualization pipeline...")
    visualize_from_json(
        results_json_path=INPUT_JSON,
        base_image_path=BASE_IMAGE_DIR,
        output_dir=OUTPUT_VISUALIZATION_DIR
    )
    print(f"Pipeline finished. Visualizations saved in {OUTPUT_VISUALIZATION_DIR}")
