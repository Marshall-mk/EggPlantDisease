import os
import random
import shutil

def create_test_set(DATASET_DIR):
    # make a test set by moving 20% of the images from the sub folders of original folder to the test folder
    for folder in os.listdir(DATASET_DIR + 'train/'):
        if not os.path.exists(DATASET_DIR + 'test/' + folder + '/'):
            os.makedirs(DATASET_DIR + 'test/' + folder + '/')
        files = os.listdir(DATASET_DIR + 'train/' + folder)
        num_files = len(files)
        num_test = int(.2 * num_files)
        test_files = random.sample(files, num_test)
        for file in test_files:
            shutil.move(DATASET_DIR + 'train/' + folder + '/' + file, DATASET_DIR + 'test/' + folder + '/' + file)
        print('Moved ' + str(len(test_files)) + ' images to test folder for ' + folder)


if __name__ == "__main__":
    DATASET_DIR = '../data/'
    create_test_set(DATASET_DIR)