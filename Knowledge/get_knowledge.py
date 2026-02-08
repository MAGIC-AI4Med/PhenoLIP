import json
import csv
import pandas as pd

def parse_hpo_data(json_data):
    """
    解析HPO JSON数据并生成三个CSV文件
    """
    
    # 获取图数据
    graph = json_data["graphs"][0]
    nodes = graph["nodes"]
    edges = graph["edges"]
    
    # 创建节点ID到标签的映射
    id_to_label = {}
    node_info = {}
    
    # 解析节点信息
    for node in nodes:
        node_id = node["id"]
        label = node.get("lbl", "")
        
        # 提取定义
        definition = ""
        if "meta" in node and "definition" in node["meta"]:
            definition = node["meta"]["definition"].get("val", "")
        
        # 提取注释
        comments = ""
        if "meta" in node and "comments" in node["meta"]:
            comments = "; ".join(node["meta"]["comments"])
        
        # 存储节点信息
        id_to_label[node_id] = label
        node_info[node_id] = {
            "hpo_id": node_id,
            "label": label,
            "definition": definition,
            "comments": comments
        }
    
    # 生成第一个文件：hpo_name2concept.csv
    concept_data = []
    for node_id, info in node_info.items():
        concept_data.append([
            info["hpo_id"],
            info["label"],
            info["definition"],
            info["comments"]
        ])
    
    # 写入hpo_name2concept.csv
    with open("hpo_name2concept.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hpo_id", "label", "definition", "comments"])
        writer.writerows(concept_data)
    
    # 解析边信息，生成第二和第三个文件
    kg_data = []  # 用于hpo_kg.csv
    id_data = []  # 用于hpo_id.csv
    
    for edge in edges:
        head_id = edge["sub"]  # subject作为head
        tail_id = edge["obj"]  # object作为tail
        relation = edge["pred"]  # predicate作为关系
        
        # 获取标签
        head_label = id_to_label.get(head_id, "")
        tail_label = id_to_label.get(tail_id, "")
        
        # 添加到数据列表
        kg_data.append([head_label, tail_label, relation])
        id_data.append([head_id, tail_id])
    
    # 写入hpo_kg.csv
    with open("hpo_kg.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["head_hpo_label", "tail_hpo_label", "relation"])
        writer.writerows(kg_data)
    
    # 写入hpo_id.csv
    with open("hpo_id.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["head_hpo_id", "tail_hpo_id"])
        writer.writerows(id_data)
    
    print(f"生成的文件统计:")
    print(f"- hpo_name2concept.csv: {len(concept_data)} 个概念")
    print(f"- hpo_kg.csv: {len(kg_data)} 个关系")
    print(f"- hpo_id.csv: {len(id_data)} 个关系")
    
    return concept_data, kg_data, id_data

# 使用示例
def main():
    # 这里假设您的JSON数据已经加载到变量json_data中
    # 如果是从文件读取，可以这样做：
    with open("data/hpo/hp.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    

    # 解析数据并生成CSV文件
    concept_data, kg_data, id_data = parse_hpo_data(json_data)
    
    # 显示前几行数据作为验证
    print("\nhpo_name2concept.csv 前几行:")
    df_concept = pd.DataFrame(concept_data, columns=["hpo_id", "label", "definition", "comments"])
    print(df_concept.head())
    
    print("\nhpo_kg.csv 前几行:")
    df_kg = pd.DataFrame(kg_data, columns=["head_hpo_label", "tail_hpo_label", "relation"])
    print(df_kg.head())
    
    print("\nhpo_id.csv 前几行:")
    df_id = pd.DataFrame(id_data, columns=["head_hpo_id", "tail_hpo_id"])
    print(df_id.head())

    df_concept.to_csv('code_stage6/KAD/Rarevisual/Data/df_concept.csv',index=False)
    df_id.to_csv('code_stage6/KAD/Rarevisual/Data/df_id.csv',index=False)
    df_kg.to_csv('code_stage6/KAD/Rarevisual/Data/df_kg.csv',index=False)

if __name__ == "__main__":
    main()