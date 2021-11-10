import cv2, os
import numpy as np

def segment(image_path, number ,folder_name):
    ## read image
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## threshold 
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    hh, ww = thresh.shape

    ## make bottom 2 rows black where they are white the full width of the image
    thresh[hh-3:hh, 0:ww] = 0

    ## get bounds of white pixels
    white = np.where(thresh==255)
    xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])

    ## segment the image at the bounds adding back the two blackened rows at the bottom
    img_seg = img[ymin:ymax+3, xmin:xmax]

    ## save resulting masked image
    image_name = folder_name + '_' + str(number) + '.jpg'
    new_path = '../Dataset_segmented/' + folder_name + '/' + image_name 
    
    cv2.imwrite(new_path, img_seg)
    ## display result
    # cv2.imshow("thresh", thresh)
    # cv2.imshow("image segment", img_seg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()