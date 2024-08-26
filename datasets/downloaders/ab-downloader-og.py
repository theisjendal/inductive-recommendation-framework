import argparse
import os
import subprocess
from gzip import GzipFile
from zipfile import ZipFile

from loguru import logger

from shared.utility import valid_dir, download_progress

BASE_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
FILE_LIST = ['reviews_Books.json.gz', 'meta_Books.json.gz', 'reviews_Books_5.json.gz', 'meta_Books_5.json.gz']
parser = argparse.ArgumentParser()
parser.add_argument('--out_path', nargs=1, type=valid_dir, default='../amazon-book',
                    help='path to store downloaded data')

def download(path):
    path = os.path.join(path, 'data')
    os.makedirs(path, exist_ok=True)

    for file in FILE_LIST:
        out_name = file.rsplit('.', 1)[0]
        download_progress(BASE_URL + file, path, file)

        out_path = os.path.join(path, out_name)
        if not os.path.isfile(out_path):
            arg = f'pv {os.path.join(path, file)} | gzip -d -c > {out_path}'
            logger.info(arg)
            p = subprocess.Popen(arg, shell=True)

            p.wait()
        else:
            logger.warning('Skipping extraction, already exist.')


if __name__ == '__main__':
    args = parser.parse_args()
    download(args.out_path)
