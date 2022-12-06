import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import numpy as np
from datetime import datetime

from utils import draw_segmentation_mask

if __name__ == "__main__":
    now = datetime.now()

    test_video_path = "../test_video_segmentation/test_videos/GH010767.MP4"
    out_video_path = "../test_video_segmentation/results/test_video_result_" + now.strftime("%Y%m%d-%H%M%S") + ".avi"
    model_path = "../saved_models/unet_model.h5"

    unet_model = keras.models.load_model(model_path)

    cap = cv.VideoCapture(test_video_path)

    if not cap.isOpened():
        print("Cannot open test video!")
        exit()

    fps = cap.get(cv.CAP_PROP_FPS)

    FRAME_HEIGHT = 256 + 8*16
    FRAME_WIDTH = 256 + 12*16

    fourcc = cv.VideoWriter_fourcc(*"XVID")
    out = cv.VideoWriter(out_video_path, fourcc, fps, (FRAME_WIDTH,  FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = cv.resize(frame, [FRAME_WIDTH, FRAME_HEIGHT])

            mask = unet_model.predict(np.expand_dims(frame/255.0, axis=0), verbose=0)[0]
            mask = tf.one_hot(tf.argmax(mask, axis=2), depth=3)
            mask = cv.resize(np.array(mask), [FRAME_WIDTH, FRAME_HEIGHT], interpolation=cv.INTER_NEAREST)

            new_frame = draw_segmentation_mask(frame, mask, alpha=0.5)

            out.write(cv.cvtColor(new_frame, cv.COLOR_RGB2BGR))

    cap.release()
    out.release()
    cv.destroyAllWindows()