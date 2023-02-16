import os
import pandas as pd
import pickle
from underthesea import text_normalize
import numpy as np
from collections import Counter
import torch
import config
from sklearn import preprocessing
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchvision.transforms as transforms
import cv2

model_bert = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')


def process(x):
    x = str(x)
    x = x.lower()
    x = x.strip()
    x = text_normalize(x)
    return x


def processing_split(x):
    x = x.split(',')
    if '' in x: x.remove('')
    return list(map(process, x))


def t(x, remove_actor):
    def remove_other(t):
        if t in remove_actor:
            t = 'other'
        return t

    result = list(map(remove_other, x))
    result = np.array(result)
    return np.unique(result)


def clear_label(labels):
    def rename_label(x):
        if x in ['trinh thám', 'phá án', 'hình sự']:
            return 'tội phạm'
        elif x in ['bí ẩn', 'huyền huyễn', 'siêu nhiên', 'siêu năng lực', 'giả tưởng', 'thần thoại']:
            return 'huyền bí'
        elif x in ['thanh xuân', 'trường học']:
            return 'huyền bí'
        elif x == 'võ thuật':
            return 'hành động'
        elif x == 'viễn tây':
            return 'phiêu lưu'
        elif x == 'hồi hộp':
            return 'giật gân'
        else:
            return x

    labels = list(map(rename_label, labels))
    remove_list = []
    for x in labels:
        if x not in config.label_define:
            remove_list.append(x)
    if len(remove_list) > 0:
        for i in remove_list:
            labels.remove(i)
    return labels


def clear_country(x):
    if x not in ['mỹ', 'trung quốc', 'hàn quốc', 'việt nam', 'thái lan']:
        if x in ['china', 'hồng kông', 'đài loan']:
            return 'trung quốc'
        elif x in ['korea']:
            return 'hàn quốc'
        elif x in ['canada', 'chile', 'us', 'uruguay', 'brazil', 'mỹ', 'mexico', 'argentina']:
            return 'âu mỹ'
        else:
            return 'other'
    else:
        return x


path_data = 'data/json'

data_frames = []
for i in os.listdir(path_data):
    path_json = os.path.join(path_data, i)
    df = pd.read_json(path_json)
    data_frames.append(df)
data = pd.concat(data_frames)

data['film_bo'].fillna(0, inplace=True)
data['film_bo'] = data['film_bo'].astype(int)
data.drop_duplicates(subset=['series_name'], inplace=True)
data.replace('', np.nan, inplace=True)
data.dropna(subset=['raw_category_name'], inplace=True)
data['description'] = data['series_name'] + ' ' + data['description']
data.drop(columns=['release_year', 'url', 'film_bo'], inplace=True)
data['director_name'].fillna('Unknow', inplace=True)
data.reset_index(inplace=True, drop=True)

for i in data:

    if i != 'series_name':
        data[i] = data[i].apply(lambda x: process(x))
data['actor_name'] = data['actor_name'].apply(lambda x: processing_split(x))
data['raw_category_name'] = data['raw_category_name'].apply(lambda x: processing_split(x))
x = []
for i in data['actor_name']:
    x.extend(i)
remove_actor = []

counter_actor = Counter(x)
for i in counter_actor.keys():
    if counter_actor[i] < 2: remove_actor.append(i)

data['actor_name'] = data['actor_name'].apply(lambda x: t(x, remove_actor))
data['raw_category_name'] = data['raw_category_name'].apply(lambda x: clear_label(x))
data['country']=data['country'].apply(lambda x: clear_country(x))


le = preprocessing.LabelBinarizer()
le.fit(data['director_name'])

countrys = preprocessing.LabelBinarizer()
cao = countrys.fit_transform(data['country'])
print(data.country.value_counts())
actor = MultiLabelBinarizer()
actor.fit(data['actor_name'])

x = data[['series_name', 'description', 'country', 'director_name',
          'actor_name']]

X_train, X_test, y_train, y_test = train_test_split(x, data['raw_category_name'], test_size=0.2)
mlb = MultiLabelBinarizer()
ymlb_train = mlb.fit_transform(y_train)
ymlb_test = mlb.transform(y_test)


def encode_data(X, Y, train=True):
    if train:
        path_save = 'data/train/'
    else:
        path_save = 'data/test/'
    for index, t in enumerate(X.iterrows()):
        _, i = t
        item = dict()
        split = i['series_name'].split(',')
        name = split[0]
        name = name.replace('/', '_')
        path = os.path.join(config.image_path, name.replace(':', "_").strip() + '.png')

        if not os.path.exists(path):
            name = split[0] + ',' + split[1]
            path = os.path.join(config.image_path, name.replace(':', "_").strip() + '.png')
        img = cv2.imread(path)
        img = cv2.resize(img, (256, 256))
        transform = transforms.ToTensor()
        tensor = transform(img)

        embeddings = model_bert.encode(tokenize(i['description']), convert_to_tensor=True)

        x1 = countrys.transform([i['country']])
        x2 = le.transform([i['director_name']])
        x3 = actor.transform([i['actor_name']])

        item['images'] = tensor
        item['description'] = embeddings
        item['country'] = torch.from_numpy(x1)
        item['director_name'] = torch.from_numpy(x2)
        item['actor_name'] = torch.from_numpy(x3)
        item['label'] = torch.from_numpy(Y[index])
        with open(path_save + '({}).pickle'.format(index), 'wb') as handle:
            pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)


encode_data(X_train, ymlb_train)
encode_data(X_test, ymlb_test, train=False)
