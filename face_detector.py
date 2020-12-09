import os
import cv2


# confirm path
# print(cv2)
# '/Users/tiger/opt/anaconda3/envs/dl/lib/python3.7/site-packages/cv2/cv2.cpython-37m-darwin.so'

# count the number of images
DIR = './images'
N = sum(os.path.isfile(os.path.join(DIR, name)) for name in os.listdir(DIR))

# define cascade classifier
cascade = cv2.CascadeClassifier('/Users/tiger/opt/anaconda3/envs/dl/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')


for i in range(N):
    # detect smile
    img = cv2.imread('./images/image' + str(i+1) + '.jpg')
    height = img.shape[0]
    width = img.shape[1]

    face = cascade.detectMultiScale(img)

    # make bounding boxes
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 200), thickness=2)

    # show the image
    resized_img = cv2.resize(img,(width//2, height//2))
    cv2.imshow('image', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save the image
    cv2.imwrite('./results/result' + str(i+1) + '.jpg', img)

