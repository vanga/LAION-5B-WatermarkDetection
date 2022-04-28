import torch
import argparse
from inference import inference
from pathlib import Path

def main():
    """
    Launch inference using torch spawn with number of processes equal to the number of devices.
    """
    args = parse_args()
    print(f"Available devices: {torch.cuda.device_count()}")
    torch.multiprocessing.spawn(inference, nprocs=torch.cuda.device_count(), args=(args,))


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls', type=str, help='file urls', required=True)
    parser.add_argument('--output-dir', type=Path, help='Output directory where result parquet files will be writtern', default="./output")
    parser.add_argument('--output-num-samples', type=int, help='Number of samples per output metadata file', default=1e6)
    parser.add_argument('--output-columns',type=str, help='list of columns to be included in output metadata file', default="image_filename")
    parser.add_argument('--include-hash-of', type=str, help='comman separated values of column names to include as a hash in the output metadata file. Ex: url, text')
    parser.add_argument('--num-workers', type=int, help="Number of sub processes to use for data loading (pytorch dataloader workers)", default=8)
    parser.add_argument('--num-samples', type=int, help='Total number of samples, only used for progress calculation', default=1024)
    parser.add_argument('--batch-size', type=int, help='samples per batch', default=256)
    parser.add_argument('--bucket-dir', type=str, help='output bucket directory') # not useful unless vpc endpoint is also configurable
    return parser.parse_args()

if __name__ == '__main__':
    main()