import torch

def print_gpu_info():
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    print(f"CUDA 버전: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU 개수: {torch.cuda.device_count()}")
        print(f"현재 GPU: {torch.cuda.current_device()}")
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    print_gpu_info()