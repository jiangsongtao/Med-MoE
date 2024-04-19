import os
import json
import shutil
import tarfile
import argparse
import subprocess
import hashlib
from multiprocessing import Pool
import multiprocessing as mp
from urllib.error import HTTPError
import urllib.request
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='./llava_med_image_urls.jsonl')
parser.add_argument('--pmc_output_path', type=str, default='/data/pmc_articles/')
parser.add_argument('--images_output_path', type=str, default='/data/images/')
parser.add_argument('--remove_pmc', action='store_true', default=True, help='remove pmc articles after image extraction')
parser.add_argument('--cpus', type=int, default=-1, help='number of cpus to use in multiprocessing (default: all)')
args = parser.parse_args()

input_data = []
with open(args.input_path) as f:
    for line in f:
        input_data.append(json.loads(line))

def download_func(idx):
    sample = input_data[idx]
    try:
        output_tar_file = os.path.join(args.pmc_output_path, os.path.basename(sample['pmc_tar_url']))
        if not os.path.exists(output_tar_file):  # 检查文件是否已经存在
            urllib.request.urlretrieve(sample['pmc_tar_url'], output_tar_file)
        
            tar = tarfile.open(output_tar_file, "r:gz")
            tar.extractall(args.pmc_output_path)
            tar.close()
        
        src = os.path.join(args.pmc_output_path, sample['image_file_path'])
        dst = os.path.join(args.images_output_path, sample['pair_id']+'.jpg')
        
        if not os.path.exists(dst):  # 检查文件是否已经存在
            shutil.copyfile(src, dst)  
        
        if args.remove_pmc and os.path.exists(output_tar_file):  # 如果需要删除已下载的压缩文件
            os.remove(output_tar_file)
            shutil.rmtree(os.path.join(args.pmc_output_path, str(os.path.basename(sample['pmc_tar_url']))).split('.tar.gz')[0]+'/')
    except HTTPError as e:
        print(f"HTTPError: {e.code} - {e.reason}. Skipping {sample['pmc_tar_url']}")
    except Exception as e:
        print(e)
if args.cpus == -1:
    cpus = mp.cpu_count()
else:
    cpus = args.cpus

pool = Pool(cpus)

# Using tqdm to display download progress
with tqdm(total=len(input_data)) as pbar:
    for _ in tqdm(pool.imap_unordered(download_func, range(0, len(input_data)))):
        pbar.update()

print("Download completed.")
