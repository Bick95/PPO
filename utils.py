import os
import json
import random
import string
from datetime import datetime


def get_unique_save_path(len_rand_str: int = 10):
    now = datetime.now()
    today_string = now.strftime("%Y_%m_%d__%H_%M_%S")
    random_string = ''.join(random.choice(string.ascii_letters) for i in range(len_rand_str))

    return today_string + '__' + random_string


def save(args: json, attribute: str, saver):
    # Check if saving-path is provided. Otherwise don't save
    if hasattr(args, attribute):
        # Check if provided path exists, if not create it
        save_path = '/'.join(getattr(args, attribute).split('/')[:-1])
        print("Save path: ", save_path)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # Save policy net
        saver(getattr(args, attribute))
