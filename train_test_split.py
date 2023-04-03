import glob
import os
import random
import shutil

imgs = os.listdir('imgs')
src_folder = './imgs/'
train_folder = './finetuning/train/'
val_folder = './finetuning/val/'

with open ('memes.txt', 'r') as f:
    memes = f.readlines()

for meme in memes:
    meme_name = meme.split('/')[-1].strip()
    pattern = src_folder + meme_name + '*'
    files = glob.glob(pattern)
    if len(files) == 1:
        os.mkdir(train_folder + meme_name)
        os.mkdir(val_folder + meme_name)
        file_name = os.path.basename(files[0])
        shutil.move(files[0], train_folder + meme_name + '/' + file_name)
    elif len(files) > 1:
        os.mkdir(train_folder + meme_name)
        os.mkdir(val_folder + meme_name)
        random.shuffle(files)
        train = files[:int(len(files)*0.8)]
        val = files[int(len(files)*0.8):]
        for file in train:
            # extract file name form file path
            file_name = os.path.basename(file)
            meme_name 
            shutil.move(file, train_folder + meme_name + '/' + file_name)
        for file in val:
            # extract file name form file path
            file_name = os.path.basename(file)
            shutil.move(file, val_folder + meme_name + '/' + file_name)

