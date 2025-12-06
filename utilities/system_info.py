import torch

print(f"Torch version: {torch.__version__}")

def print_cpu_info():
  print("CPU Info:")
  print(f"  Number of CPUs: {torch.get_num_threads()}")
  print(f"  MKL Enabled: {torch.backends.mkl.is_available()}")
  print(f"  OpenMP Enabled: {torch.backends.openmp.is_available()}")
  print()

def print_cuda_info():
  print("CUDA Info:")
  print(f"  CUDA Available: {torch.cuda.is_available()}")
  if torch.cuda.is_available():
    print(f"  Number of CUDA Devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
      print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
      print(f"    Capability: {torch.cuda.get_device_capability(i)}")
      print(f"    Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
      print(f"    Memory Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
  print()

if __name__ == "__main__":
  print_cpu_info()
  print_cuda_info()