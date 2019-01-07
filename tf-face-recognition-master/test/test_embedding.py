import numpy as np
from PIL import Image

from tensorface import embedding
from tensorface.const import FACE_PIC_SIZE
from tensorface.detection import img_to_np


def test_embedding_rand():
    embedding.embedding(np.random.randint(0, 254, (FACE_PIC_SIZE, FACE_PIC_SIZE, 3)))


def test_embedding():
    pic = Image.open('./sumit.png')
    f_pic = pic.crop((0, 0, FACE_PIC_SIZE, FACE_PIC_SIZE))
    f_np = img_to_np(f_pic)
    # crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC)
    e = embedding.embedding(f_np)
    e.shape == embedding.embedding_size()


def test_embedding_size():
    # [batch_size, height, width, 3]
    assert embedding.embedding_size() == 512


def test_input_shape():
    assert embedding.input_shape() == (1, 2)

'''def initTrainingSet():
    train_files = [f for f in os.listdir(TRAIN_EXAMPLES_DIR) if f.endswith('.png')]
    print(train_files)
    for f in  train_files:
        name, size, num = f.split(".")[0].split("_")[1:]
        img = Image.open(os.path.join(TRAIN_EXAMPLES_DIR, f))
        learn_from_examples(name, img, int(num), int(size))
    print("train complete!")

initTrainingSet()'''
