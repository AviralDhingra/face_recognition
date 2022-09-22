import numpy as np
from IPython.display import display
from PIL import Image, ImageDraw

import face_recognition
from list import convert


def learn():
    names, images = convert()
    known_face_encodings = []

    for i in range(len(images)):
        image = face_recognition.load_image_file(f'img_learn/{images[i]}')
        image_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(image_encoding)

    known_face_names = names
    print('Learned encoding for', len(known_face_encodings), 'images.')
    return known_face_encodings, known_face_names
