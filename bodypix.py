import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import cv2
from matplotlib import pyplot as plt
import numpy as np

# create model
bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


_, init_f = cap.read()
width, height, _ = init_f.shape
bg_img = cv2.imread("bg.jpeg")
bg_img = bg_img[:width, :height, :]

while True:
    ret, frame = cap.read()

    # prediction
    result = bodypix_model.predict_single(frame)
    mask = result.get_mask(threshold=0.5).numpy().astype(np.uint8)
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)

    neg = np.add(mask, -1)
    inverse = np.where(neg==-1, 1, neg).astype(np.uint8)
    masked_bg = cv2.bitwise_and(bg_img, bg_img, mask=inverse)
    final = cv2.add(masked_image, masked_bg)

    cv2.imshow("BodyPix", final)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()