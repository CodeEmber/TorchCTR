'''
Author       : wyx-hhhh
Date         : 2024-04-25
LastEditTime : 2024-04-25
Description  : 
'''
import lmdb
import pickle
from PIL import Image
from generate_lmdb import LMDB_Image
from utils.file_utils import get_file_path


class LMDB2Jpg():

    def __init__(self) -> None:
        self.lmdb_path = get_file_path(['data', 'pixelrec', 'images_lmdb'])
        self.image_save_path = get_file_path(['data', 'pixelrec', 'images'], add_sep_affter=True)

    def __enter__(self):
        self.lmdb_env = lmdb.open(self.lmdb_path, readonly=False, meminit=False, map_async=True)
        self.lmdb_txn = self.lmdb_env.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lmdb_env.close()

    def lmdb2jpg(self):
        keys_data = self.lmdb_txn.get(b'__keys__')
        keys = pickle.loads(keys_data)
        i = 0
        for key in keys:
            data = self.lmdb_txn.get(key)
            if data is not None:
                img_obj = pickle.loads(data)
                image_array = img_obj.get_image()
                img = Image.fromarray(image_array)
                img.save(self.image_save_path + key.decode('ascii') + '.jpg')

    def get_key_image(self, key):
        data = self.lmdb_txn.get(key)
        if data is not None:
            img_obj = pickle.loads(data)
            image_array = img_obj.get_image()
            img = Image.fromarray(image_array)
            return img


if __name__ == '__main__':
    with LMDB2Jpg() as lmdb2jpg:
        lmdb2jpg.lmdb2jpg()
        img = lmdb2jpg.get_key_image(b'i225967')
        img.show()
