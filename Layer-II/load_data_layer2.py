import cv2, os, imutils
import numpy as np
from imblearn.over_sampling import SMOTE
from Preprocess import preparation
from sklearn import preprocessing

train_data_dir = '../Dataset_holdout/train'
test_data_dir = '../Dataset_holdout/test'

def preparation(image):
    try:
        # Grayscale conversion
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ############################################# Image Enhancement ##############################################################
        # Power-law Tranformation
        img = np.array(255*(img/255)**0.5,dtype='uint8')
        # 2-dimesional Gaussian filter
        img = cv2.GaussianBlur(img,(3,3),0)

        ############################################# Feature Extraction ##############################################################
        # Histogram Analysis
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        # L2-normalization
        if imutils.is_cv2():
            hist = cv2.normalize(hist)
        else:
            cv2.normalize(hist, hist)

        return hist.flatten()
    except Exception as x:
        print(str(x))

############################################################ Load data #########################################################################      

def load_training_data():
    # Load training images
    labels = os.listdir(train_data_dir)
    total = len(labels)
    X_train = []
    Y_train = []

    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    for label in labels:
        i = 0
        image_names_train = os.listdir(os.path.join(train_data_dir, label))
        total = len(image_names_train)

        ############################################## Labeled for COVID-19 class ########################################################################
        if label == 'COVID-19' :
            j = 0
            print(label,j)
            for image_name in image_names_train:
                try:
                    if image_name != 'Thumbs.db':
                        img = cv2.imread((os.path.join(train_data_dir, label, image_name)))
                        img = cv2.resize(img, (500, 500))

                        hist = preparation(img)
                        X_train.append(hist)
                        Y_train.append(j)


                except Exception as e:
                    print(str(e))
                
                if i % 100 == 0:
                    print('Done: {0}/{1} images'.format(i, total))
                i += 1

        ############################################# Labeled for NON-COVID-19 class ########################################################################
        elif label == 'Virus' or label == 'Bacterail':
            j = 1
            print(label,j)
            for image_name in image_names_train:
                try:
                    if image_name != 'Thumbs.db':
                        img = cv2.imread((os.path.join(train_data_dir, label, image_name)))
                        img = cv2.resize(img, (500, 500))

                        hist = preparation(img)
                        X_train.append(hist)
                        Y_train.append(j)


                except Exception as e:
                    print(str(e))
                
                if i % 100 == 0:
                    print('Done: {0}/{1} images'.format(i, total))
                i += 1
            
    print(i)
    print('Loading done.')

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    le = preprocessing.LabelEncoder()
    le.fit(Y_train)

    Y_train = le.transform(Y_train)
    Y_train = Y_train.reshape(Y_train.shape[0], -1)

    ############################################# Balancing Data with SMOTE Technique ########################################################################
    sm = SMOTE(random_state=42)
    X_train, Y_train = sm.fit_resample(X_train,Y_train)

    ############################################# Save data to .NPY File ########################################################################
    np.save('x_train_layer2.npy', X_train)
    np.save('y_train_layer2.npy', Y_train)

    return X_train, Y_train



def load_testing_data():
    # Load training images
    labels = os.listdir(test_data_dir)
    total = len(labels)
    X_test = []
    Y_test = []

    print('-' * 30)
    print('Creating testing images...')
    print('-' * 30)
    for label in labels:
        i = 0
        image_names_test = os.listdir(os.path.join(test_data_dir, label))
        total = len(image_names_test)

        ############################################## Labeled for COVID-19 class ########################################################################
        if label == 'COVID-19':
            j = 0
            print(label,j)
            for image_name in image_names_test:
                try:
                    if image_name != 'Thumbs.db':
                        img = cv2.imread((os.path.join(test_data_dir, label, image_name)))
                        img = cv2.resize(img, (500, 500))

                        hist = preparation(img)
                        X_test.append(hist)
                        Y_test.append(j)


                except Exception as e:
                    print(str(e))
                
                if i % 100 == 0:
                    print('Done: {0}/{1} images'.format(i, total))
                i += 1

        ############################################# Labeled for Non-COVID-19 class ########################################################################
        elif label == 'Virus' or label == 'Bacterail':
            j = 1
            print(label,j)
            for image_name in image_names_test:
                try:
                    if image_name != 'Thumbs.db':
                        img = cv2.imread((os.path.join(test_data_dir, label, image_name)))
                        img = cv2.resize(img, (500, 500))

                        hist = preparation(img)
                        X_test.append(hist)
                        Y_test.append(j)


                except Exception as e:
                    print(str(e))
                
                if i % 100 == 0:
                    print('Done: {0}/{1} images'.format(i, total))
                i += 1
            
    print(i)
    print('Loading done.')

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    le = preprocessing.LabelEncoder()
    le.fit(Y_test)

    Y_test = le.transform(Y_test)
    Y_test = Y_test.reshape(Y_test.shape[0], -1)

    ############################################# Balancing Data with SMOTE Technique ########################################################################
    sm = SMOTE(random_state=42)
    X_test, Y_test = sm.fit_resample(X_test,Y_test)

    ############################################# Save data to .NPY File ########################################################################
    np.save('x_test_layer2.npy', X_test)
    np.save('y_test_layer2.npy', Y_test)

    return X_test, Y_test

X_train,y_train = load_training_data()
X_test,y_test = load_testing_data()  
