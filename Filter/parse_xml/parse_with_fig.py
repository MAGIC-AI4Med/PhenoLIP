from lxml import etree
import os
import json
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import sys
import argparse

# 定义处理XML文件的函数
def node_text(node):
    result = ""
    for text in node.itertext():
        result = result + text
    return result

def extract_figure_info(fig_node):
    """提取图片的详细信息"""
    figure_info = {}
    
    # 提取图片ID
    figure_id = fig_node.get('id')
    if figure_id:
        figure_info['id'] = figure_id
    
    # 提取图片位置属性
    position = fig_node.get('position')
    if position:
        figure_info['position'] = position
    
    # 提取label（如Figure 1）
    label_node = fig_node.find('.//label')
    if label_node is not None:
        figure_info['label'] = node_text(label_node).strip()
    
    # 提取caption（图片说明）
    caption_info = {}
    caption_node = fig_node.find('.//caption')
    if caption_node is not None:
        # 提取caption中的title
        title_node = caption_node.find('.//title')
        if title_node is not None:
            caption_info['title'] = node_text(title_node).strip()
        
        # 提取caption中的所有段落
        paragraphs = []
        for p_node in caption_node.findall('.//p'):
            p_text = node_text(p_node).strip()
            if p_text:
                paragraphs.append(p_text)
        
        if paragraphs:
            caption_info['paragraphs'] = paragraphs
        
        # 如果没有单独的title和paragraphs，提取整个caption的文本
        if not caption_info:
            caption_text = node_text(caption_node).strip()
            if caption_text:
                caption_info['text'] = caption_text
    
    if caption_info:
        figure_info['caption'] = caption_info
    
    # 提取graphic信息（图片文件路径等）
    graphics = []
    for graphic_node in fig_node.findall('.//graphic'):
        graphic_info = {}
        
        # 提取href属性（图片文件路径）
        href = graphic_node.get('{http://www.w3.org/1999/xlink}href')
        if href:
            graphic_info['href'] = href
        
        # 提取其他属性
        for attr_name, attr_value in graphic_node.attrib.items():
            if attr_name not in ['{http://www.w3.org/1999/xlink}href']:
                # 移除命名空间前缀
                clean_attr_name = attr_name.split('}')[-1] if '}' in attr_name else attr_name
                graphic_info[clean_attr_name] = attr_value
        
        if graphic_info:
            graphics.append(graphic_info)
    
    if graphics:
        figure_info['graphics'] = graphics
    
    return figure_info

def iternode(node, df, father_tag):
    if node.tag == 'article':
        df['info']['article-type'] = node.get("article-type")
    if node.tag == 'article-id':
        df['info'][node.get('pub-id-type')] = node.text  
    if node.tag == 'article-title':
        df['text'] =  df['text'] + '\n' + node_text(node) + '\n'  
    if node.tag == 'abstract':
        df['abstract']['start'] = len(df['text']) 
    if node.tag == 'xref':
        type = node.get('ref-type')
        id = node.get('rid')
        temp_dd = {'id': id, 'start':len(df['text'])}
    
    # 处理figure节点
    if node.tag == 'fig':
        figure_info = extract_figure_info(node)
        df['figure'].append(figure_info)
        return df  # 处理完figure后直接返回，避免重复处理子节点
    
    current_tag = father_tag+'.'+node.tag   
    if 'body' in current_tag or 'abstract' in current_tag:    
        if 'table-wrap' not in current_tag and 'fig' not in current_tag:
            if 'title' in current_tag:
                df['text'] =  df['text'] + '\n\n'
            if node.text != None:
                df['text'] =  df['text'] +  node.text 

    if 'back' in current_tag:
        return df
    
    for subnode in node.getchildren():
        try:
             df = iternode(subnode, df, father_tag=current_tag)
        except:
             continue

    if node.tag == 'xref':
        type = node.get('ref-type')
        id = node.get('rid')
        temp_dd['end'] = len(df['text'])
        if type == 'fig':
            df['img_ref'].append(temp_dd)
        if type == 'bibr':
            df['bibref'].append(temp_dd)
        if type == 'table':
            df['tab_ref'].append(temp_dd)

    if 'body' in current_tag or 'abstract' in current_tag:    
        if 'table-wrap' not in current_tag and 'fig' not in current_tag:
            if node.tail != None:
                df['text'] =  df['text'] + node.tail
            if 'title' in current_tag:
                df['text'] =  df['text'] + '\n\n'

    if node.tag == 'abstract':
        df['abstract']['end'] = len(df['text']) 
    return df

def preprocess(parser, xml_path, save_path):
    parser = etree.XMLParser()
    root = etree.parse(xml_path, parser).getroot()
    # 在df中添加figure字段
    df = {'info': {}, 'text': '', 'img_ref': [], 'tab_ref': [], 'bibref': [], 'abstract': {}, 'figure': []}
    df = iternode(root, df, father_tag='')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(df, f, ensure_ascii=False, indent=2)

def process_single_file(args):
    input_file_path, output_directory, input_directory = args
    # 计算相对路径，保持目录结构
    relative_path = os.path.relpath(input_file_path, input_directory)
    relative_dir = os.path.dirname(relative_path)
    
    # 在输出目录中创建对应的子目录
    output_subdir = os.path.join(output_directory, relative_dir)
    os.makedirs(output_subdir, exist_ok=True)
    
    # 构建输出文件路径，将xml后缀改为json
    output_filename = os.path.basename(input_file_path).replace('.xml', '.json')
    output_file_path = os.path.join(output_subdir, output_filename)

    # 调用 preprocess 函数处理数据
    parser = etree.XMLParser()
    try:
        preprocess(parser, input_file_path, output_file_path)
        return 1
    except Exception as e:
        print(f"处理文件 {input_file_path} 时出错: {e}")
        return 0

def process_json_files(input_directory, output_directory, max_workers=None):
    """
    遍历输入目录下的所有xml文件，使用多线程并行处理后保存到输出目录，保持原有目录结构
    """
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    # 收集所有XML文件
    xml_files = []
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    # xml_files = xml_files[:5]

    # 设置最大线程数，默认为CPU核心数
    max_workers = 32
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    total_cnt = 0
    # 使用 ThreadPoolExecutor 进行并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 准备任务列表
        tasks = [(xml_file, output_directory, input_directory) for xml_file in xml_files]
        # 使用tqdm显示进度条
        for _ in tqdm(executor.map(process_single_file, tasks), total=len(tasks)):
            total_cnt += _

    print(f'总共处理了 {total_cnt} 个文件')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process XML files in a directory to JSON.')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory containing XML files')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for JSON files')
    args = parser.parse_args()

    input_directory = args.input_dir
    output_directory = args.output_dir
    process_json_files(input_directory, output_directory)