from vllm import LLM
import os
import torch
import subprocess

def print_gpu_info():
    print("=== GPU信息 ===")
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"总显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"已分配显存: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"已缓存显存: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"可用显存: {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / 1024**3:.2f} GB")
    else:
        print("CUDA不可用")
    
    # 使用nvidia-smi获取更详细信息
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("\n=== nvidia-smi显存信息 ===")
            print("GPU | 名称 | 总显存(MB) | 已用(MB) | 可用(MB)")
            print("-" * 60)
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 5:
                    gpu_id, name, total, used, free = parts[0], parts[1], parts[2], parts[3], parts[4]
                    print(f"{gpu_id:3} | {name:20} | {total:8} | {used:7} | {free:7}")
    except FileNotFoundError:
        print("nvidia-smi不可用")

if __name__ == '__main__':
    # 打印GPU信息
    print_gpu_info()
    
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    print("\n=== 开始初始化LLM ===")
    print("正在加载模型，这可能需要一些时间...")
    
    try:
        llm = LLM(
            model="/mnt/petrelfs/liangcheng/models/qwen3-14b",
            # model="/mnt/petrelfs/liangcheng/models/qwen3-30b-a3b",
            gpu_memory_utilization=0.95,
            max_model_len=16384,
            max_num_seqs=2
        )
        
        print("模型加载成功！")
        print("\n=== 模型加载后的显存使用情况 ===")
        print_gpu_info()
        
        # 在这里添加你的其他代码
        outputs = llm.generate("Please Introduce LLM for me.")

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            
    except Exception as e:
        print(f"初始化LLM时出错: {e}")
        print("\n=== 错误发生后的显存使用情况 ===")
        print_gpu_info()