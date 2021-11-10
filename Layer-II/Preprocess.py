import cv2, imutils
import numpy as np

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