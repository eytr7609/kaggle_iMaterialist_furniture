from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import numpy as np
import os
import pandas as pd


def test():
    path = os.getcwd() + '/test'
    test_date = os.listdir(path)
    sample_submission = pd.read_csv(os.getcwd() + "/sample_submission.csv")
    model = load_model('vgg16_model_10000.h5')
    x1 = []
    ids = []
    for i in range(0, len(test_date)):
        imgpath = os.path.join(path, test_date[i])
        img = load_img(imgpath, target_size=(224, 224))
        x = img_to_array(img)
        x1.append(x)
        ids.append(os.path.splitext(test_date[i])[0])

    print('Test image load complete')
    x1 = np.asarray(x1)
    result = model.predict(x1)
    print('Predict complete')
    for i in range(0, len(result)):
        m = max(result[i])
        id = [i for i, j in enumerate(result[i]) if j == m]
        sample_submission.loc[sample_submission['id'] == int(ids[i]), 'predicted'] = id[0] + 1
    
    sample_submission.to_csv("submit.csv", index=False)


if __name__ == "__main__":
    test()
