import torch
import transformers
import datasets
import pandas
import rank_bm25

print("All imports successful!")

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))