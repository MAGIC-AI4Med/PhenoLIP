import os
import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
# --- 1. 配置与常量定义 ---

# 默认配置
DEFAULT_PMC_LIST_FILE = 'data/pmc_paper_list.json'
BASE_PATH = "/mnt/petrelfs/liangcheng/RareVisual/" 
DEFAULT_MODEL_PATH = "/mnt/petrelfs/liangcheng/models/qwen3-14b"

# <<< FIX #1: 使用我们最终确定的、功能最强大的Prompt模板 >>>
# ==============================================================================
# --- 最终版PROMPT: 批量处理子图并提取结构化信息 (通用格式示例) ---
# ==============================================================================
PROMPT_TEMPLATE = """
You are a world-class expert in medical literature AI analysis, specializing in extracting structured information from research papers and images. Your task is to process multiple sub-figures from a single image composite and generate a single, valid JSON object containing detailed, structured data for each.

You are given the following information:

1.  **Full Paper Text:**
    ---
    {full_text}
    ---

2.  **Main Caption (for the whole figure):**
    ---
    {main_caption}
    ---

3.  **Sub-figure Chunks to Process (Identifier: Caption):**
    ---
    {sub_figure_chunks_formatted}
    ---

**Your Goal:**
For EACH sub-figure chunk provided above, you must extract and generate a structured object containing three key pieces of information: an enhanced description, the imaging modality, and a list of observed phenotype names. Finally, compile all these objects into a single parent JSON object.

**Instructions:**

1.  **Identify Common Context:** First, read the "Full Paper Text" and "Main Caption" to understand the shared context (e.g., patient condition, primary diagnosis, general imaging techniques).
2.  **Process Each Sub-figure:** For each entry in "Sub-figure Chunks to Process", create a structured object with the following three fields:
    a.  **`enhanced_description` (string):**
        *   Seamlessly merge the common context with the specific findings from that sub-figure's caption chunk.
        *   The description **MUST NOT** contain any leading identifiers like "A,", "B,", "1)", etc. It must be pure, descriptive prose.
        *   Focus only on information describing what is visually present.
    b.  **`modality` (string):**
        *   Determine the single best-fit imaging modality for the sub-figure.
        *   You **MUST** choose one from this specific list: `["MRI", "CT", "X-ray", "Pathology", "Microscopy", "Ultrasound", "Natural Image", "Other"]`.
    c.  **`phenotypes` (list of strings):**
        *   Identify all clinical phenotypes or significant abnormal findings described in the text that are visible in the sub-figure.
        *   Provide a list of strings, where each string is the **common name** of a phenotype (e.g., "Vestibular schwannoma").
        *   If no specific phenotypes are clearly identifiable for a sub-figure, provide an empty list `[]`.
3.  **Format the Final Output:**
    *   Your entire output **MUST** be a single, valid JSON object and nothing else. Do not add any text before or after the JSON.
    *   The top-level JSON object should use the sub-figure identifiers (e.g., 'box_D', 'box_A') as keys.
    *   The value for each key must be the structured object described in step 2, containing the `enhanced_description`, `modality`, and `phenotypes` fields.

**Example:**

---
**EXAMPLE INPUT (`sub_figure_chunks_formatted`):**
- "identifier_1": "Original caption for the first sub-figure..."
- "identifier_2": "Original caption for the second sub-figure..."
---
**EXPECTED JSON OUTPUT (Format Only):**
```json
{{
  "identifier_1": {{
    "enhanced_description": "A detailed, clean, and synthesized description for the first sub-figure, integrating common context and specific findings, goes here.",
    "modality": "Chosen modality from the list (e.g., 'MRI')",
    "phenotypes": [
      "Name of Phenotype A found in sub-figure 1",
      "Name of Phenotype B found in sub-figure 1"
    ]
  }},
  "identifier_2": {{
    "enhanced_description": "A detailed, clean, and synthesized description for the second sub-figure goes here.",
    "modality": "Chosen modality from the list (e.g., 'CT')",
    "phenotypes": [
      "Name of Phenotype C found in sub-figure 2"
    ]
  }}
}}
```

---
**Now, process the provided information and generate the complete, nested JSON object.** /no_think
"""

# --- 2. 辅助函数 ---

def load_full_text_map(list_file_path):
    print(f"Loading PMC paper list from {list_file_path}...")
    pmc_map = {}
    with open(list_file_path, 'r') as f:
        try:
            paper_paths = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            content = f.read().strip()
            if content.startswith('[') and content.endswith(']'):
                content = content[1:-1]
            paper_paths = [p.strip().strip('"').strip(',') for p in content.split('\n') if p.strip()]

    for path in paper_paths:
        pmc_id = os.path.basename(path).replace('.json', '')
        pmc_map[pmc_id] = path
    print(f"Created map for {len(pmc_map)} PMC articles.")
    return pmc_map

def load_full_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 假设全文是以段落列表形式存储
            if 'paragraphs' in data and isinstance(data['paragraphs'], list):
                 return " ".join([p.get('text', '') for p in data['paragraphs'] if p.get('text')])
            return data.get("text", "") # 兼容旧格式
    except FileNotFoundError:
        print(f"Warning: Full text file not found at {file_path}")
        return "" # 返回空字符串而不是None，避免后续错误
    except Exception as e:
        print(f"Error reading or parsing {file_path}: {e}")
        return ""

def create_prompt_with_length_check(tokenizer, full_text, main_caption, sub_figure_chunks_formatted, max_tokens):
    """
    创建prompt并检查长度，如果超出则截取原文
    """
    # 计算除了full_text之外的固定部分的token数
    template_without_fulltext = PROMPT_TEMPLATE.replace("{full_text}", "")
    fixed_content = template_without_fulltext.format(
        main_caption=main_caption,
        sub_figure_chunks_formatted=sub_figure_chunks_formatted
    )
    
    # 获取固定部分的token数
    fixed_tokens = len(tokenizer.encode(fixed_content))
    
    # 预留一些空间给响应和安全边际
    response_reserve = 1000  # 为响应预留的token数
    safety_margin = 200     # 安全边际
    
    # 计算full_text可用的最大token数
    available_tokens_for_fulltext = max_tokens - fixed_tokens - response_reserve - safety_margin
    
    if available_tokens_for_fulltext <= 0:
        print("Warning: Fixed prompt parts are too long, using minimal full text")
        truncated_full_text = full_text[:500]  # 使用前500个字符作为最小保证
    else:
        # 检查full_text的token数
        full_text_tokens = tokenizer.encode(full_text)
        
        if len(full_text_tokens) <= available_tokens_for_fulltext:
            # 原文不需要截取
            truncated_full_text = full_text
        else:
            # 需要截取原文
            truncate_ratio = available_tokens_for_fulltext / len(full_text_tokens)
            truncate_char_count = int(len(full_text) * truncate_ratio)
            truncated_full_text = full_text[:truncate_char_count]
            
            print(f"Full text truncated: {len(full_text)} -> {len(truncated_full_text)} chars "
                  f"(ratio: {truncate_ratio:.3f})")
    
    # 生成最终prompt
    final_prompt = PROMPT_TEMPLATE.format(
        full_text=truncated_full_text,
        main_caption=main_caption,
        sub_figure_chunks_formatted=sub_figure_chunks_formatted
    )
    
    # 最终检查
    final_tokens = len(tokenizer.encode(final_prompt))
    print(final_tokens)
    if final_tokens > max_tokens:
        print(f"Warning: Final prompt still exceeds max tokens: {final_tokens} > {max_tokens}")
    
    return final_prompt

def parse_args():
    parser = argparse.ArgumentParser(description='Process medical literature with VLLM')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to input align results JSON file')
    parser.add_argument('--start-index', type=int, required=True,
                        help='Start index for processing')
    parser.add_argument('--end-index', type=int, required=True,
                        help='End index for processing')
    parser.add_argument('--pmc-list-file', type=str, default=DEFAULT_PMC_LIST_FILE,
                        help='Path to PMC paper list JSON file')
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to VLLM model')
    parser.add_argument('--output-dir', type=str, default='output/augment',
                        help='Output directory for results')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.95,
                        help='GPU memory utilization for VLLM')
    parser.add_argument('--max-model-len', type=int, default=32768,
                        help='Maximum model length')
    parser.add_argument('--max-num-seqs', type=int, default=8,
                        help='Maximum number of sequences')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Maximum tokens for generation')
    parser.add_argument('--max-input-tokens', type=int, default=28000,
                        help='Maximum input tokens (should be less than max-model-len)')
    return parser.parse_args()

# --- 3. 主执行逻辑 ---

if __name__ == '__main__':
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 输出文件路径
    output_file = os.path.join(args.output_dir, f'qwen3_structured_subfig_data_{args.start_index}_{args.end_index}.json')
    
    print(f"Processing range: {args.start_index} to {args.end_index}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {output_file}")

    # 初始化tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("Tokenizer loaded.")

    pmc_to_path_map = load_full_text_map(args.pmc_list_file)
    with open(args.input_file, 'r') as f:
        align_data = json.load(f)
    
    # 获取指定范围的数据
    total_records = len(align_data)
    start_idx = max(0, args.start_index)
    end_idx = min(total_records, args.end_index)
    
    if start_idx >= total_records:
        print(f"Start index {start_idx} is beyond total records {total_records}. Nothing to process.")
        # 创建空文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        exit(0)
    
    align_data_chunk = align_data[start_idx:end_idx]
    print(f"Processing {len(align_data_chunk)} records from index {start_idx} to {end_idx}")

    enhanced_results = []
    print("Initializing VLLM...")
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs
        # tensor_parallel_size=... # 如果需要多卡，请设置
    )
    print("VLLM Initialized.")

    # 定义采样参数，这对于JSON输出很重要
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        max_tokens=args.max_tokens, 
        top_p=0.8, 
        top_k=20
    )
    
    prompts_to_process = []
    metadata_for_prompts = []

    # --- 第一阶段：准备所有prompts ---
    print("Preparing prompts...")
    for i, paper_item in tqdm(enumerate(align_data_chunk)):
        pmc_id = paper_item.get("pmc_id")
        if not pmc_id:
            continue
        
        full_text_path = pmc_to_path_map.get(f"PMC{pmc_id}")
        if not full_text_path:
            print(f"Warning: No path found for PMC{pmc_id}. Skipping.")
            continue

        full_text = load_full_text(full_text_path)
        if not full_text:
            print(f"Warning: Empty full text for PMC{pmc_id}. Skipping.")
            continue

        for fig_item in paper_item.get("crawled_info", []):
            main_caption = fig_item.get('caption', '')
            image_path = fig_item.get('downloaded_main_image')
            figure_id = fig_item.get('id')
            alignments = fig_item.get('alignments', {}).get('align_list', [])

            if not all([main_caption, image_path, figure_id, alignments]):
                continue

            # <<< FIX #2: 正确准备模型输入字符串 >>>
            sub_figure_chunks_to_process = []
            valid_align_items = []
            for align_item in alignments:
                sub_chunk = align_item.get('caption_chunk')
                bbox_id = align_item.get('bbox_id')
                if sub_chunk and bbox_id and bbox_id != 'unknown':
                    # 使用json.dumps来正确处理引号等特殊字符
                    sub_figure_chunks_to_process.append(f'- {json.dumps(bbox_id)}: {json.dumps(sub_chunk)}')
                    valid_align_items.append(align_item)
            
            if not sub_figure_chunks_to_process:
                continue

            sub_figure_chunks_formatted = "\n".join(sub_figure_chunks_to_process)

            # 使用新的长度检查函数创建prompt
            final_prompt = create_prompt_with_length_check(
                tokenizer=tokenizer,
                full_text=full_text,
                main_caption=main_caption,
                sub_figure_chunks_formatted=sub_figure_chunks_formatted,
                max_tokens=args.max_input_tokens
            )
            
            prompts_to_process.append(final_prompt)
            # 保存元数据，以便后续匹配结果
            metadata_for_prompts.append({
                "pmc_id": pmc_id,
                "figure_id": figure_id,
                "image_path": image_path,
                "original_main_caption": main_caption,
                "valid_align_items": valid_align_items
            })

    if not prompts_to_process:
        print("No valid prompts found in the specified range. Creating empty output file.")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        exit(0)

    # --- 第二阶段：批量生成 ---
    print(f"Processing {len(prompts_to_process)} prompts in a batch...")
    # <<< FIX #3: 正确调用llm.generate，并传递变量 >>>
    outputs = llm.generate(prompts_to_process, sampling_params)
    print("Batch generation complete.")

    # --- 第三阶段：处理输出 ---
    for i, output in enumerate(outputs):
        metadata = metadata_for_prompts[i]
        generated_text = output.outputs[0].text.strip()
        
        # <<< FIX #4 & #5: 解析JSON输出并按新结构保存 >>>
        try:
            # 清理模型可能输出的前后代码块标记
            generated_text = generated_text.replace('<think>','').replace('</think>','').replace('```json','').replace('```','')
            structured_data = json.loads(generated_text)

            # 遍历JSON结果，为每个子图创建一条记录
            for bbox_id, sub_fig_data in structured_data.items():
                enhanced_results.append({
                    "pmc_id": metadata["pmc_id"],
                    "figure_id": metadata["figure_id"],
                    "bbox_id": bbox_id,
                    "image_path": metadata["image_path"],
                    # "original_main_caption": metadata["original_main_caption"],
                    # 从sub_fig_data中提取信息
                    "enhanced_description": sub_fig_data.get("enhanced_description"),
                    "modality": sub_fig_data.get("modality"),
                    "phenotypes": sub_fig_data.get("phenotypes", [])
                })
        except (json.JSONDecodeError, TypeError) as e:
            print(f"--- ERROR: Failed to parse JSON for PMC {metadata['pmc_id']}, Figure {metadata['figure_id']} ---")
            print(f"Error: {e}")
            print("Model output was:")
            print(generated_text)
            print("-----------------------------------------------------------------")

    # --- 第四阶段：保存结果 ---
    print(f"\n--- Saving {len(enhanced_results)} structured sub-figure entries to {output_file} ---")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_results, f, indent=4, ensure_ascii=False)
    
    print(f"Processing complete. Results saved to {output_file}")
