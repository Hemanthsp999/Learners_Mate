import torch

gpu_bytes = torch.cuda.get_device_properties(0).total_memory
gpu_memory_gb = round(gpu_bytes / (2**30))

print(f"GPU memory availabel {gpu_memory_gb} GB")
