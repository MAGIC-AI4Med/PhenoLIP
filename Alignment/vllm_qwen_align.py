import os
import json
import traceback
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

PROMPT_TEMPLATE="""
系统角色（system）
你是"科研医学影像图文对齐"专家。请将上传的目标检测可视化图中的检测框与给定caption进行子图-子caption级别的精确对齐。

用户角色（user）
【任务描述】

输入：
1. 目标检测可视化图：图像包含多个检测框，每个检测框中央有红色长方形标记，圆圈内显示检测框标识符（格式为'box_X'，其中X可能是字母或数字）. 如果途中没有检测框和对应的标识符，说明没有检测到
2. 包含对图像内容的描述英文caption："{caption_text}"

重要说明：
- 图像中可能存在原始的子图标记（如A、B、C、a 、b 、c 、1 、2等），这些是图像原有的标识
- 红色方框内的'box_Z','box_Y',,'box_X'等是我们需要对齐的检测框标识符，请务必区分这两者
- 检测框标识符'box_Z'中的'Z'与原始子图标记不对应

任务要求：
    1. 仔细识别每个圆圈内的检测框标识符（'box_X'格式）
    2. 观察每个检测框在图像中的具体位置
    3. 分析caption中的描述内容
    4. 根据检测框的实际位置和内容，将其与caption中最匹配的子图描述进行对齐

对齐原则：
- 以检测框的实际解剖位置和指向为准，而非字母标识的表面对应
- 如果检测框指向相同的解剖结构或描述区域，可对应同一caption段落
- 无法确定匹配关系时标记为"unknown"

输出格式：
仅输出json格式的对齐结果, 禁止输出非json的解释内容：
```json
[
    {"bbox_id": "检测框标识符", "caption_chunk": "对应的原文描述片段"},
    {"bbox_id": "检测框标识符", "caption_chunk": "对应的原文描述片段"},
    ...
]
```
"""
PROMPT_TEMPLATE_origin = """
系统角色（system）
你是"科研医学影像图文对齐"专家。请将上传的目标检测可视化图中的检测框与给定caption进行精确对齐。

用户角色（user）
【任务描述】

输入：
1. 目标检测可视化图：图像包含多个检测框，每个检测框中央有红色圆圈标记，圆圈内显示检测框标识符（格式为'box_X'，其中X可能是字母或数字）
2. 包含对图像内容的描述英文caption：<caption>

重要说明：
- 图像中可能存在原始的子图标记（如A、B、C、D等），这些是图像原有的标识
- 红色圆圈内的'box_X'是我们需要对齐的检测框标识符，请务必区分这两者
- 检测框标识符'box_X'中的X与原始子图标记可能不对应

任务要求：
    1. 仔细识别每个红色圆圈内的检测框标识符（'box_X'格式）
    2. 观察每个检测框在图像中的具体位置和所指向的解剖结构
    3. 分析caption中的描述内容
    4. 根据检测框的实际位置和指向内容，将其与caption中最匹配的描述进行对齐

对齐原则：
- 以检测框的实际解剖位置和指向为准，而非字母标识的表面对应
- 如果检测框指向相同的解剖结构或描述区域，可对应同一caption段落
- 无法确定匹配关系时标记为"unknown"
- 如果存在不确定的，"Confidence"字段输出为"False"；如果确定则为"True"。

输出格式：
仅输出json格式的对齐结果：
```json
{
    "align_list":[
        {"bbox_id": "检测框标识符", "caption_chunk": "对应的原文描述片段"},
        {"bbox_id": "检测框标识符", "caption_chunk": "对应的原文描述片段"},
        ...
    ],
    "Confidence": "True/False"
}
```
"""

class AlignmentProcessor:
    def __init__(self, model_path: str, processor_path: str = None, 
                tensor_parallel_size: int = 2, max_num_seqs: int = 2,
                gpu_memory_utilization: float = 0.95):
        """
        初始化对齐处理器
        
        Args:
            model_path: 模型路径
            processor_path: 处理器路径，默认与model_path相同
            tensor_parallel_size: 张量并行大小
            max_num_seqs: 最大并发序列数
            gpu_memory_utilization: GPU内存利用率
        """
        if processor_path is None:
            processor_path = model_path
            
        # 设置vLLM环境变量
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        
        print("Loading vLLM model...")
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"image": 10, "video": 10},
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_model_len=4096,
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.000001,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=1024,
            stop_token_ids=[],
        )
        
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(processor_path)
        
    def prepare_batch_inputs(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        准备批量输入数据
        
        Args:
            batch_data: 包含图像路径和caption的数据列表
            
        Returns:
            准备好的vLLM输入列表
        """
        llm_inputs = []
        
        for item in batch_data:
            vis_fig_path = item['vis_fig_path']
            caption = item['caption']
            
            # 检查图像文件是否存在
            if vis_fig_path is None:
                vis_fig_path = item['downloaded_main_image']
            if not os.path.exists(vis_fig_path):
                item['error'] = f"Image file not found: {vis_fig_path}"
                continue
                
            prompt = PROMPT_TEMPLATE.replace('{caption_text}', str(caption))
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": vis_fig_path,
                            "min_pixels": 256 * 28 * 28,
                            "max_pixels": 2800 * 28 * 28,
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            try:
                # 应用聊天模板
                formatted_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                
                # 处理视觉信息
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True
                )
                
                # 准备多模态数据
                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs
                
                llm_input = {
                    "prompt": formatted_prompt,
                    "multi_modal_data": mm_data,
                    "mm_processor_kwargs": video_kwargs,
                }
                
                llm_inputs.append({
                    'llm_input': llm_input,
                    'original_item': item
                })
                
            except Exception as e:
                item['error'] = f"Error preparing input: {str(e)}"
                print(f"Error preparing input for {vis_fig_path}: {e}")
                
        return llm_inputs
    
    def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理数据
        
        Args:
            batch_data: 包含图像路径和caption的数据列表
            
        Returns:
            处理结果列表
        """
        # 准备输入
        llm_inputs = self.prepare_batch_inputs(batch_data)
        
        if not llm_inputs:
            return [{'alignments': [], 'error': 'No valid inputs'} for _ in batch_data]
        
        try:
            # 批量推理
            outputs = self.llm.generate(
                [item['llm_input'] for item in llm_inputs], 
                self.sampling_params
            )
            
            # 处理输出
            results = []
            output_idx = 0
            
            for item in batch_data:
                if 'error' in item:
                    results.append({'alignments': [], 'error': item['error']})
                    continue
                    
                if output_idx < len(outputs):
                    try:
                        generated_text = outputs[output_idx].outputs[0].text
                        alignment = self._parse_alignment_result(generated_text)
                        results.append({'alignments': alignment})
                    except Exception as e:
                        results.append({
                            'alignments': [], 
                            'error': f"Error parsing output: {str(e)}"
                        })
                    output_idx += 1
                else:
                    results.append({'alignments': [], 'error': 'No output generated'})
                    
            return results
            
        except Exception as e:
            print(f"Batch processing error: {e}")
            traceback.print_exc()
            return [{'alignments': [], 'error': f'Batch processing failed: {str(e)}'} 
                for _ in batch_data]
    

    def _parse_alignment_result(self, text: str) -> Dict[str, Any]:
        """
        解析对齐结果
        
        Args:
            text: 模型生成的文本
            
        Returns:
            解析后的对齐结果
        """
        try:
            json_part = extract_json_from_text(text)
                
            alignment = json.loads(json_part)
            if isinstance(alignment, dict):
                return alignment
            elif isinstance(alignment, list):
                return {"align_list":alignment}
            else:
                return {}
        except Exception as e:
            print(f"[JSON解析错误] {e}")
            print(f"[无法解析的返回内容]:\n{text}\n")
            return {
                'error': f'JSON parsing error: {str(e)}',
                'unparsed_text': text
            }
def extract_json_from_text(text):
        """
        从模型输出中提取JSON内容，处理各种复杂情况
        """
        import re
        
        text = text.strip()
        
        # 方法1: 优先查找```json...```模式
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        if matches:
            # 如果有多个匹配，选择最长的那个（通常是最完整的）
            longest_match = max(matches, key=len)
            if longest_match.strip():
                return longest_match.strip()
        else :
            pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
            sec_matches = re.findall(pattern, text, re.DOTALL)
            if sec_matches:
                longest_match = max(sec_matches, key=len)
                if longest_match.strip():
                    return longest_match.strip()
        # 方法6: 最后兜底，返回原文本
        return text
        
def collect_figure_items(data: List[Dict], start_index: int, end_index: int) -> List[Dict[str, Any]]:
    """
    收集需要处理的图片项目 - 兼容新旧数据结构
    
    Args:
        data: 输入数据
        start_index: 开始索引
        end_index: 结束索引
        
    Returns:
        需要处理的图片项目列表
    """
    figure_items = []
    sub_data = data[start_index:end_index]
    
    for item_idx, item in enumerate(sub_data):
        for fig_idx, fig_item in enumerate(item.get('crawled_info', [])):
            caption = fig_item.get('enhanced_captions', '')
            
            # 检查是否有检测结果
            detection_results = fig_item.get('processing_results', [])
            if not detection_results:
                continue
            
            vis_fig_path = fig_item['processing_results'].get('visualization_path','')
            
            
                
            figure_items.append({
                'vis_fig_path': vis_fig_path,
                'caption': caption,
                'item_idx': item_idx,
                'fig_idx': fig_idx,
                'pmc_id': item.get('pmc_id', ''),
                'fig_id': fig_item.get('id', ''),
                'detection_count': len(detection_results)
            })
            
    return figure_items

def main():
    parser = argparse.ArgumentParser(description='处理PubMed文章图文对齐 - vLLM版本')
    parser.add_argument('--input-file', type=str, required=True, help='输入JSON文件路径')
    parser.add_argument('--start-index', type=int, default=0, help='开始索引')
    parser.add_argument('--end-index', type=int, required=True, help='结束索引')
    parser.add_argument('--output-dir', type=str, default='output', help='输出目录')
    parser.add_argument('--output-suffix', type=str, default='', help='输出文件名后缀')
    parser.add_argument('--batch-size', type=int, default=2, help='批处理大小')
    parser.add_argument('--tensor-parallel-size', type=int, default=2, help='张量并行大小')
    parser.add_argument('--max-num-seqs', type=int, default=1, help='最大并发序列数')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.95, help='GPU内存利用率')
    args = parser.parse_args()

    output_fn = os.path.join(
        args.output_dir,
        f"align_res_{args.start_index}_{args.end_index}{args.output_suffix}.json"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = "SubFig_Detection/pretrained_model/qwen2.5vl72b"
    
    # 初始化处理器
    processor = AlignmentProcessor(
        model_path=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    print(f"Loading data: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Total records in input: {len(data)}")

    # 截取当前任务需要处理的部分
    sub_data = data[args.start_index:args.end_index]
    print(f"Processing records from {args.start_index} to {args.end_index} ({len(sub_data)})")

    # 收集所有需要处理的图片项目
    figure_items = collect_figure_items(data, args.start_index, args.end_index)
    print(f"Total figure items to process: {len(figure_items)}")
    
    # 打印一些统计信息
    if figure_items:
        detection_counts = [item['detection_count'] for item in figure_items]
        print(f"Detection boxes per figure - Min: {min(detection_counts)}, Max: {max(detection_counts)}, Avg: {sum(detection_counts)/len(detection_counts):.1f}")

    if not figure_items:
        print("No figure items found to process.")
        with open(output_fn, 'w', encoding='utf-8') as fout:
            json.dump(sub_data, fout, ensure_ascii=False, indent=2)
        return

    # 批量处理
    print("Starting alignment...")
    results = {}
    
    # 按批次处理
    for i in tqdm(range(0, len(figure_items), args.batch_size), desc="Processing batches"):
        batch = figure_items[i:i+args.batch_size]
        batch_results = processor.process_batch(batch)
        
        # 存储结果
        for item, result in zip(batch, batch_results):
            key = (item['item_idx'], item['fig_idx'])
            results[key] = result
            
            if 'error' in result:
                print(f"[错误] {item['pmc_id']} {item['fig_id']}: {result['error']}")
            else:
                # 打印成功处理的信息
                align_count = len(result.get('alignments', {}).get('align_list', []))
                print(f"[成功] {item['pmc_id']} {item['fig_id']}: {align_count} alignments found")

    # 将结果分配回原始数据结构
    for item_idx, item in enumerate(sub_data):
        for fig_idx, fig_item in enumerate(item.get('crawled_info', [])):
            key = (item_idx, fig_idx)
            if key in results:
                fig_item['alignments'] = results[key]['alignments']
            else:
                fig_item['alignments'] = []

    # 保存结果
    with open(output_fn, 'w', encoding='utf-8') as fout:
        json.dump(sub_data, fout, ensure_ascii=False, indent=2)
    print(f"本批次完成，结果已保存在: {output_fn}")

if __name__ == '__main__':
    main()
