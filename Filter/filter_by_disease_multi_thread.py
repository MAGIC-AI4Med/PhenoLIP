import os
import json
import re
import argparse
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class DiseaseFilter:
    def __init__(self, concept_file: str):
        """
        初始化疾病筛选器
        
        Args:
            concept_file: 罕见病概念到ID映射文件路径
        """
        self.disease_concepts = self._load_disease_concepts(concept_file)
        self.disease_abbreviations: Set[str] = set()
        self.disease_full_names: Set[str] = set()
        self._classify_diseases()
        
    def _load_disease_concepts(self, file_path: str) -> Dict[str, str]:
        """加载罕见病概念映射"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _classify_diseases(self):
        """
        将疾病名称根据是否包含空格，分类为缩写和全称。
        """
        print("正在对疾病名称进行分类...")
        for disease_name in self.disease_concepts.keys():
            if ' ' in disease_name.strip():
                self.disease_full_names.add(disease_name)
            else:
                self.disease_abbreviations.add(disease_name)
        print(f"分类完成：{len(self.disease_abbreviations)}个缩写，{len(self.disease_full_names)}个全称。")

    def _preprocess_text(self, text: str) -> str:
        """预处理文本，标准化格式"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _match_abbreviations(self, text: str) -> List[Tuple[str, str, float]]:
        """
        规则1: 对疾病缩写进行精确匹配（单词级别）。
        """
        matches = []
        processed_text = self._preprocess_text(text)
        text_words = set(processed_text.split())
        for disease_name in self.disease_abbreviations:
            disease_words = set(disease_name.split())
            if disease_words and disease_words.issubset(text_words):
                matches.append((disease_name, self.disease_concepts[disease_name], 1.0))
        return matches

    def _match_full_names(self, text: str) -> List[Tuple[str, str, float]]:
        """
        规则2: 对疾病全称进行全文匹配（短语级别）。
        """
        matches = []
        processed_text = self._preprocess_text(text)
        for disease_name in self.disease_full_names:
            processed_disease = self._preprocess_text(disease_name)
            if processed_disease and processed_disease in processed_text:
                matches.append((disease_name, self.disease_concepts[disease_name], 1.0))
        return matches
    
    def filter_article(self, article_data: Dict) -> Dict:
        """
        筛选文章是否与罕见病相关
        """
        text_content = article_data.get('text', '')
        # text_content = '\n'.join(['\n'.join(caption['paragraphs']) for caption in article_data['caption']])
        if not text_content:
            return {
                'is_disease_related': False,
                'matched_diseases': [],
                'total_matches': 0,
                'highest_confidence': 0.0
            }
            
        all_matches = []
        abb_matches = self._match_abbreviations(text_content)
        full_matches = self._match_full_names(text_content)
        all_matches.extend(abb_matches)
        all_matches.extend(full_matches)

        final_abb_matches = [(disease_name, disease_id) for disease_name, disease_id, confidence in abb_matches]
        final_full_matches = [(disease_name, disease_id) for disease_name, disease_id, confidence in full_matches]
        
        return {
            'is_disease_related': len(all_matches) > 0,
            'abb_matches': final_abb_matches,
            'full_matches': final_full_matches,
            'total_matches': len(all_matches),
        }

def process_single_article(file_path: str, filter_obj: DiseaseFilter) -> Tuple[bool, Dict]:
    """
    处理单个文章文件
    
    Args:
        file_path: 文章文件路径
        filter_obj: DiseaseFilter 实例
    
    Returns:
        Tuple[bool, Dict]: (是否成功, 结果字典或错误信息)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            article_data = json.load(f)
        
        result = filter_obj.filter_article(article_data)
        if result['is_disease_related']:
            result['file_name'] = file_path
        return True, result
    except Exception as e:
        return False, {'file_name': os.path.basename(file_path), 'error': str(e)}

def process_articles_multithreaded(start_idx: int, end_idx: int, file_list_path: str, concept_file_path: str, max_workers: int = None) -> None:
    """
    使用多线程处理文章文件
    
    Args:
        start_idx: 起始索引
        end_idx: 终止索引
        file_list_path: 文件列表路径
        concept_file_path: 疾病概念文件路径
        max_workers: 最大线程数，默认为 CPU 核心数
    """
    # 初始化筛选器
    filter_obj = DiseaseFilter(concept_file_path)
    
    with open(file_list_path, 'r', encoding='utf-8') as f:
        json_files = json.load(f)[start_idx:end_idx]
    
    # 设置最大线程数
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    filtered_results = []
    total_success = 0
    
    # 使用 ThreadPoolExecutor 进行并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 准备任务元组
        task_tuples = [(file, filter_obj) for file in json_files]
        
        # 使用 tqdm 显示进度条
        for success, result in tqdm(
            executor.map(lambda t: process_single_article(t[0], t[1]), task_tuples),
            total=len(json_files),
            desc="处理文章"
        ):
            if success and result['is_disease_related']:
                filtered_results.append(result)
            if success:
                total_success += 1
            else:
                print(f"处理失败: {result['file_name']} - {result['error']}")

    print(f"共处理 {len(json_files)} 个文章，成功 {total_success} 个，发现 {len(filtered_results)} 篇与罕见病相关")
    final_res = {
        'total_articles': len(json_files),
        'related_articles': len(filtered_results),
        'filtered_results': filtered_results
    }
    # 保存筛选结果
    with open(f'output/filter_with_disease/res_{start_idx}_{end_idx}.json', 'w', encoding='utf-8') as f:
        json.dump(final_res, f, ensure_ascii=False, indent=2)

def main():
    """
    主函数，使用 argparse 解析命令行参数
    """
    parser = argparse.ArgumentParser(description="筛选与罕见病相关的文章")
    parser.add_argument(
        '--start_idx', 
        type=int, 
        required=True, 
        help='起始索引'
    )
    parser.add_argument(
        '--end_idx', 
        type=int, 
        required=True, 
        help='终止索引'
    )
    parser.add_argument(
        '--file_list_path', 
        type=str,  
        default='data/pmc_paper_list.json',
        help='文件列表路径'
    )
    parser.add_argument(
        '--concept_file_path', 
        type=str, 
        default='data/Orpha/orpha_concept2id.json', 
        help='疾病概念文件路径'
    )
    parser.add_argument(
        '--max_workers', 
        type=int, 
        default=16, 
        help='最大线程数，默认为 CPU 核心数'
    )
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs('output', exist_ok=True)
    
    process_articles_multithreaded(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        file_list_path=args.file_list_path,
        concept_file_path=args.concept_file_path,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main()