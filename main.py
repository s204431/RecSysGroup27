import torch
import torch.nn as nn

def set_seed(seed):
    torch.manual_seed(seed)

    # Hvis du bruger GPU (CUDA), så sæt seed for alle GPU-enheder
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For at sætte seed på alle CUDA enheder

    # For at få deterministiske resultater på CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Slå off auto-tuning af CUDA-kernels for deterministisk output


def main():
    set_seed(42)
























if __name__ == '__main__':
    main()