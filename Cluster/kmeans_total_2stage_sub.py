import os
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '32'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pandas as pd
import tensorflow as tf
from PIL import Image
import shutil
from tqdm import tqdm
import time

def load_features(npz_path):
    """加载保存的特征文件"""
    print(f"Loading features from {npz_path}...")
    
    data = np.load(npz_path)
    features = data['features']
    paths = data['paths']
    
    print(f"Loaded {features.shape[0]} features with dimension {features.shape[1]}")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Paths array shape: {paths.shape}")
    
    return features, paths

def explore_features(features):
    """探索特征的基本统计信息"""
    print("\n=== Feature Statistics ===")
    print(f"Mean: {features.mean():.4f}")
    print(f"Std: {features.std():.4f}")
    print(f"Min: {features.min():.4f}")
    print(f"Max: {features.max():.4f}")
    print(f"Feature dimension: {features.shape[1]}")
    
    # 检查是否有NaN或无穷值
    print(f"NaN values: {np.isnan(features).sum()}")
    print(f"Inf values: {np.isinf(features).sum()}")
    
    return features

def preprocess_features(features, normalize=True):
    """预处理特征"""
    # 去除NaN和无穷值
    mask = np.isfinite(features).all(axis=1)
    clean_features = features[mask]
    print(f"Removed {features.shape[0] - clean_features.shape[0]} samples with invalid values")
    
    if normalize:
        # 标准化特征
        scaler = StandardScaler()
        clean_features = scaler.fit_transform(clean_features)
        print("Features normalized")
    
    return clean_features, mask

def two_level_kmeans(features, n_clusters_level1=20, n_clusters_level2=20):
    """
    执行两级K-means聚类
    
    Args:
        features: 特征矩阵 (n_samples, n_features)
        n_clusters_level1: 第一级聚类数量（大簇）
        n_clusters_level2: 第二级聚类数量（每个大簇内的小簇）
    
    Returns:
        level1_labels: 第一级聚类标签
        level2_labels: 第二级聚类标签
        cluster_info: 聚类信息字典
    """
    n_samples = features.shape[0]
    print(f"\n=== Starting Two-Level K-means Clustering ===")
    print(f"Total samples: {n_samples:,}")
    print(f"Level 1: {n_clusters_level1} clusters")
    print(f"Level 2: {n_clusters_level2} clusters per level-1 cluster")
    print(f"Total clusters: {n_clusters_level1 * n_clusters_level2}")
    
    # 第一级聚类 - 使用MiniBatchKMeans处理大数据
    print("\n--- Level 1 Clustering ---")
    start_time = time.time()
    
    if n_samples > 50000:
        # 对于大数据集使用MiniBatchKMeans
        kmeans_level1 = MiniBatchKMeans(
            n_clusters=n_clusters_level1,
            random_state=42,
            batch_size=1000,
            verbose=1
        )
    else:
        kmeans_level1 = KMeans(
            n_clusters=n_clusters_level1,
            random_state=42,
            verbose=1
        )
    
    level1_labels = kmeans_level1.fit_predict(features)
    level1_time = time.time() - start_time
    print(f"Level 1 clustering completed in {level1_time:.2f} seconds")
    
    # 统计第一级聚类结果
    unique_l1, counts_l1 = np.unique(level1_labels, return_counts=True)
    print(f"Level 1 cluster sizes: min={counts_l1.min()}, max={counts_l1.max()}, mean={counts_l1.mean():.1f}")
    
    # 第二级聚类 - 对每个大簇分别聚类
    print("\n--- Level 2 Clustering ---")
    level2_labels = np.zeros(n_samples, dtype=int)
    cluster_info = {}
    
    for cluster_id in tqdm(range(n_clusters_level1), desc="Processing Level 1 clusters"):
        # 获取当前大簇的样本
        cluster_mask = level1_labels == cluster_id
        cluster_features = features[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        n_samples_in_cluster = cluster_features.shape[0]
        
        if n_samples_in_cluster == 0:
            continue
        
        # 调整第二级聚类数量（如果样本太少）
        actual_n_clusters = min(n_clusters_level2, n_samples_in_cluster)
        
        if n_samples_in_cluster > 1000:
            # 对于大簇使用MiniBatchKMeans
            kmeans_level2 = MiniBatchKMeans(
                n_clusters=actual_n_clusters,
                random_state=42,
                batch_size=min(100, n_samples_in_cluster),
                verbose=0
            )
        else:
            kmeans_level2 = KMeans(
                n_clusters=actual_n_clusters,
                random_state=42,
                verbose=0
            )
        
        # 执行第二级聚类
        local_labels = kmeans_level2.fit_predict(cluster_features)
        
        # 转换为全局标签（大簇ID * n_clusters_level2 + 小簇ID）
        global_labels = cluster_id * n_clusters_level2 + local_labels
        level2_labels[cluster_indices] = global_labels
        
        # 保存聚类信息
        for local_id in range(actual_n_clusters):
            global_id = cluster_id * n_clusters_level2 + local_id
            local_mask = local_labels == local_id
            cluster_info[global_id] = {
                'level1_cluster': cluster_id,
                'level2_cluster': local_id,
                'size': local_mask.sum(),
                'indices': cluster_indices[local_mask]
            }
    
    print(f"\nTotal unique clusters created: {len(cluster_info)}")
    
    return level1_labels, level2_labels, cluster_info

def save_clustering_results(features, image_paths, level1_labels, level2_labels, 
                           cluster_info, output_dir="clustering_results"):
    """
    保存聚类结果和样本展示
    
    Args:
        features: 特征矩阵
        image_paths: 图片路径
        level1_labels: 第一级聚类标签
        level2_labels: 第二级聚类标签
        cluster_info: 聚类信息
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Saving Clustering Results ===")
    
    # 1. 创建完整的聚类信息DataFrame
    metadata = []
    for i in range(len(features)):
        metadata.append({
            'index': i,
            'image_path': image_paths[i],
            'filename': os.path.basename(image_paths[i]),
            'directory': os.path.basename(os.path.dirname(image_paths[i])),
            'level1_cluster': level1_labels[i],
            'level2_cluster': level2_labels[i],
            'combined_cluster': f"L1_{level1_labels[i]:02d}_L2_{level2_labels[i]:03d}"
        })
    
    df_full = pd.DataFrame(metadata)
    
    # 保存完整的聚类信息
    full_tsv_path = os.path.join(output_dir, 'full_clustering_results.tsv')
    df_full.to_csv(full_tsv_path, sep='\t', index=False)
    print(f"Saved full clustering results to: {full_tsv_path}")
    
    # 2. 统计每个簇的信息
    cluster_stats = []
    for cluster_id, info in cluster_info.items():
        cluster_stats.append({
            'cluster_id': cluster_id,
            'level1_cluster': info['level1_cluster'],
            'level2_cluster': info['level2_cluster'],
            'size': info['size']
        })
    
    df_stats = pd.DataFrame(cluster_stats)
    stats_path = os.path.join(output_dir, 'cluster_statistics.tsv')
    df_stats.to_csv(stats_path, sep='\t', index=False)
    print(f"Saved cluster statistics to: {stats_path}")
    
    # 3. 为每个簇保存10个样本作为展示
    samples_dir = os.path.join(output_dir, 'cluster_samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    sample_metadata = []
    print("\nSaving sample images for each cluster...")
    
    for cluster_id in tqdm(cluster_info.keys(), desc="Saving cluster samples"):
        info = cluster_info[cluster_id]
        indices = info['indices']
        
        # 随机选择最多10个样本
        n_samples = min(10, len(indices))
        if n_samples > 0:
            sample_indices = np.random.choice(indices, n_samples, replace=False)
            
            # 为每个样本创建记录
            for idx in sample_indices:
                sample_metadata.append({
                    'cluster_id': cluster_id,
                    'level1_cluster': info['level1_cluster'],
                    'level2_cluster': info['level2_cluster'],
                    'sample_index': idx,
                    'image_path': image_paths[idx],
                    'filename': os.path.basename(image_paths[idx])
                })
                
                # 可选：复制样本图片到对应的簇文件夹
                cluster_sample_dir = os.path.join(samples_dir, f"cluster_{cluster_id:03d}")
                os.makedirs(cluster_sample_dir, exist_ok=True)
                if os.path.exists(image_paths[idx]):
                    shutil.copy2(image_paths[idx], cluster_sample_dir)
    
    # 保存样本元数据
    df_samples = pd.DataFrame(sample_metadata)
    samples_tsv_path = os.path.join(output_dir, 'cluster_samples.tsv')
    df_samples.to_csv(samples_tsv_path, sep='\t', index=False)
    print(f"Saved cluster samples metadata to: {samples_tsv_path}")
    
    return df_full, df_stats, df_samples

def create_sprite_image(images, sprite_path, image_size=(100, 100)):
    """创建sprite图像用于TensorBoard"""
    n_images = len(images)
    grid_size = int(np.ceil(np.sqrt(n_images)))
    
    sprite_height = grid_size * image_size[1]
    sprite_width = grid_size * image_size[0]
    sprite = Image.new('RGB', (sprite_width, sprite_height), (255, 255, 255))
    
    for i, img_path in enumerate(images):
        try:
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = img.resize(image_size, Image.Resampling.LANCZOS)
                
                row = i // grid_size
                col = i % grid_size
                x = col * image_size[0]
                y = row * image_size[1]
                
                sprite.paste(img, (x, y))
        except Exception as e:
            continue
    
    sprite.save(sprite_path)
    print(f"Sprite image saved: {sprite_path}")

def create_tensorboard_visualization(features, image_paths, level1_labels, level2_labels,
                                    log_dir="tb_clustering_logs", n_samples=5000):
    """
    创建TensorBoard Projector可视化（随机采样）
    """
    print(f"\n=== Creating TensorBoard Visualization ===")
    print(f"Randomly sampling {n_samples} points for visualization...")
    
    # 随机采样
    total_samples = len(features)
    if total_samples > n_samples:
        sample_indices = np.random.choice(total_samples, n_samples, replace=False)
        sampled_features = features[sample_indices]
        sampled_paths = image_paths[sample_indices]
        sampled_l1_labels = level1_labels[sample_indices]
        sampled_l2_labels = level2_labels[sample_indices]
    else:
        sampled_features = features
        sampled_paths = image_paths
        sampled_l1_labels = level1_labels
        sampled_l2_labels = level2_labels
    
    # 降维用于可视化
    print("Applying PCA for visualization...")
    pca = PCA(n_components=min(50, sampled_features.shape[1]))
    reduced_features = pca.fit_transform(sampled_features)
    print(f"Reduced to {reduced_features.shape[1]} dimensions")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # 清理并创建日志目录
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    # 保存特征
    features_tensor = tf.Variable(reduced_features, name='features')
    checkpoint = tf.train.Checkpoint(embedding=features_tensor)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))
    
    # 创建元数据
    metadata = []
    for i in range(len(sampled_features)):
        metadata.append({
            'index': i,
            'image_path': sampled_paths[i],
            'filename': os.path.basename(sampled_paths[i]),
            'level1_cluster': sampled_l1_labels[i],
            'level2_cluster': sampled_l2_labels[i],
            'combined': f"L1_{sampled_l1_labels[i]:02d}_L2_{sampled_l2_labels[i]:03d}"
        })
    
    df_meta = pd.DataFrame(metadata)
    metadata_path = os.path.join(log_dir, 'metadata.tsv')
    df_meta.to_csv(metadata_path, sep='\t', index=False)
    
    # 创建sprite图像
    sprite_path = os.path.join(log_dir, 'sprite.png')
    create_sprite_image(sampled_paths, 
                       sprite_path, image_size=(80, 80))
    
    # 创建projector配置
    config = f"""embeddings {{
  tensor_name: "embedding/.ATTRIBUTES/VARIABLE_VALUE"
  metadata_path: "metadata.tsv"
  sprite {{
    image_path: "sprite.png"
    single_image_dim: 80
    single_image_dim: 80
  }}
}}"""
    
    config_path = os.path.join(log_dir, 'projector_config.pbtxt')
    with open(config_path, 'w') as f:
        f.write(config)
    
    print(f"TensorBoard files created in: {log_dir}")
    print(f"Run: tensorboard --logdir {log_dir}")
    
    return log_dir

def analyze_clustering_quality(features, level1_labels, level2_labels, sample_size=10000):
    """
    分析聚类质量
    """
    print("\n=== Clustering Quality Analysis ===")
    
    # 如果数据太大，采样计算
    if len(features) > sample_size:
        print(f"Sampling {sample_size} points for quality metrics...")
        indices = np.random.choice(len(features), sample_size, replace=False)
        sampled_features = features[indices]
        sampled_l1 = level1_labels[indices]
        sampled_l2 = level2_labels[indices]
    else:
        sampled_features = features
        sampled_l1 = level1_labels
        sampled_l2 = level2_labels
    
    # 计算轮廓系数
    print("Calculating silhouette scores...")
    sil_l1 = silhouette_score(sampled_features, sampled_l1, sample_size=min(5000, len(sampled_features)))
    sil_l2 = silhouette_score(sampled_features, sampled_l2, sample_size=min(5000, len(sampled_features)))
    
    print(f"Level 1 Silhouette Score: {sil_l1:.4f}")
    print(f"Level 2 Silhouette Score: {sil_l2:.4f}")
    
    # 计算Calinski-Harabasz分数
    ch_l1 = calinski_harabasz_score(sampled_features, sampled_l1)
    ch_l2 = calinski_harabasz_score(sampled_features, sampled_l2)
    
    print(f"Level 1 Calinski-Harabasz Score: {ch_l1:.2f}")
    print(f"Level 2 Calinski-Harabasz Score: {ch_l2:.2f}")
    
    return {
        'silhouette_l1': sil_l1,
        'silhouette_l2': sil_l2,
        'calinski_harabasz_l1': ch_l1,
        'calinski_harabasz_l2': ch_l2
    }

# ================================
# 主程序
# ================================

if __name__ == "__main__":
    # 1. 加载数据
    npz_path = "/mnt/petrelfs/liangcheng/hpo_feature/sub_fig_dino_features.npz"
    features, image_paths = load_features(npz_path)
    
    # 2. 探索和预处理
    features = explore_features(features)
    clean_features, valid_mask = preprocess_features(features, normalize=True)
    valid_paths = image_paths[valid_mask]
    
    print(f"\nClean features shape: {clean_features.shape}")
    print(f"Valid paths count: {len(valid_paths)}")
    
    # 3. 执行两级K-means聚类
    level1_labels, level2_labels, cluster_info = two_level_kmeans(
        clean_features,
        n_clusters_level1=20,
        n_clusters_level2=20
    )
    
    # 4. 保存聚类结果
    output_dir = "code_stage6/subfig_cluster/two_level_clustering_results"
    df_full, df_stats, df_samples = save_clustering_results(
        clean_features,
        valid_paths,
        level1_labels,
        level2_labels,
        cluster_info,
        output_dir
    )
    
    # 5. 分析聚类质量
    quality_metrics = analyze_clustering_quality(
        clean_features,
        level1_labels,
        level2_labels,
        sample_size=10000
    )
    
    # 6. 创建TensorBoard可视化（随机5000个样本）
    tb_log_dir = create_tensorboard_visualization(
        clean_features,
        valid_paths,
        level1_labels,
        level2_labels,
        log_dir="code_stage6/subfig_cluster/two_level_tb_clustering_visualization",
        n_samples=5000
    )
    
    # 7. 打印总结
    print("\n" + "="*60)
    print("TWO-LEVEL K-MEANS CLUSTERING COMPLETED!")
    print("="*60)
    print(f"Total samples processed: {len(clean_features):,}")
    print(f"Level 1 clusters: 20")
    print(f"Level 2 clusters per L1: 20")
    print(f"Total theoretical clusters: 400")
    print(f"Actual clusters created: {len(cluster_info)}")
    print(f"\nOutput files:")
    print(f"  - Full results: {output_dir}/full_clustering_results.tsv")
    print(f"  - Statistics: {output_dir}/cluster_statistics.tsv")
    print(f"  - Sample metadata: {output_dir}/cluster_samples.tsv")
    print(f"\nQuality Metrics:")
    for key, value in quality_metrics.items():
        print(f"  - {key}: {value:.4f}")
    print(f"\nTensorBoard Visualization:")
    print(f"  1. Run: tensorboard --logdir {tb_log_dir}")
    print(f"  2. Open: http://localhost:6006")
    print(f"  3. Click 'PROJECTOR' tab to explore clusters")
    print(f"=  logdir {tb_log_dir}")
    print(f"  2. Open: http://localhost:6006")
    print(f"  3. Click 'PROJECTOR' tab to explore clusters")
    print("="*60)