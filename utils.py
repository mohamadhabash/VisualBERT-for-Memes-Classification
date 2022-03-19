from nltk import word_tokenize
from tqdm.auto import tqdm

import re
import cv2

import matplotlib.pyplot as plt


def convert_to_BGR(image_file_names, images_path):
    bgr_images = []
    for image in tqdm(image_file_names):
        try:
            img = plt.imread(f'{images_path}/{image}')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bgr_images.append(img)

        except FileNotFoundError:
            continue
    
    return bgr_images


def preprocess_text(text):
    nltk.download('punkt')

    # Remove extra whitespaces
    text = text.strip()

    # Remove twitter usernames, web addresses
    text = text = re.sub(r"#[\w\d]*|@[.]?[\w\d]*[\'\w*]*|https?:\/\/\S+\b|"r"www\.(\w+\.)+\S*|", '', text)

    # Remove html tags
    text = re.sub(re.compile('<.*?>'), ' ', text)

    # Remove unwanted characters (unicode, non-english)
    text = word_tokenize(text)
    text = ' '.join(word for word in text if word.isalpha() or word.isnumeric() or word.isalnum())
  
    return text

