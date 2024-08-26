import argparse
import os
from zipfile import ZipFile

from shared.utility import valid_dir, download_progress

BASE_URL = "https://raw.githubusercontent.com/xiangwang1223/knowledge_graph_attention_network/master/Data/amazon-book/"
FILE_LIST = ['entity_list.txt', 'item_list.txt', 'kg_final.txt.zip', 'relation_list.txt', 'test.txt', 'train.txt',
             'user_list.txt']

parser = argparse.ArgumentParser()
parser.add_argument('--out_path', nargs=1, type=valid_dir, default='../amazon-book',
                    help='path to store downloaded data')


def download(path):
    path = os.path.join(path, 'data')
    os.makedirs(path, exist_ok=True)

    for file in FILE_LIST:
        download_progress(BASE_URL + file, path, file)

    with ZipFile(os.path.join(path, 'kg_final.txt.zip')) as zf:
        zf.extractall(path)


if __name__ == '__main__':
    args = parser.parse_args()
    download(args.out_path)
