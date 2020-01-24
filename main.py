import os
import numpy as np
import tensorflow as tf
from data_reader import DataReader
from model import Model
import time
from PIL import Image, ImageDraw

DATA_DIR = "D:\\MachineLearning\\VOC\\VOCdevkit\\VOC2007"
LOG_DIR = "D:\\MachineLearning\\SSD_Transfer\\log"

save_freq = 10#50
eval_freq = 10000000

IMAGE_SIZE = [300, 300]
BATCH_SIZE = 8#16
FEATURE_SIZES = [37, 19, 10, 5, 3, 1]
NUM_ANCHORS = [4, 6, 6, 6, 4, 4]
ASPECT_RATIOS = [1, 2, 1/2, 3, 1/3]

def draw_prediction(images, classes_pred, locations_pred, default_boxies):
    drawn_images = []

    for b in range(classes_pred[0].shape[0]):
        image = Image.fromarray((images[b] * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)

        for f in range(len(classes_pred)):
            for x in range(classes_pred[f].shape[1]):
                for y in range(classes_pred[f].shape[2]):
                    for a in range(classes_pred[f].shape[3]):
                        if classes_pred[f][b, x, y, a] == 0: continue

                        cx = (locations_pred[f][b, x, y, a, 0] * default_boxies[f][b, x, y, a, 2] + default_boxies[f][b, x, y, a, 0]) * image.size[0]
                        cy = (locations_pred[f][b, x, y, a, 1] * default_boxies[f][b, x, y, a, 3] + default_boxies[f][b, x, y, a, 1]) * image.size[1]
                        wid = (np.exp(1) ** locations_pred[f][b, x, y, a, 2]) * default_boxies[f][b, x, y, a, 2] * image.size[0]
                        hei = (np.exp(1) ** locations_pred[f][b, x, y, a, 3]) * default_boxies[f][b, x, y, a, 3] * image.size[1]
                        draw.rectangle([cx - wid / 2, cy - hei / 2, cx + wid / 2, cy + hei / 2], outline="red")
                        draw.text([cx - wid / 2, cy - hei / 2], DataReader.class_to_name(None, classes_pred[f][b, x, y, a]), fill="red")

        drawn_images.append(image)

    return drawn_images

def train():
    print("\nPrepare reader")
    train_reader = DataReader(DATA_DIR, IMAGE_SIZE, BATCH_SIZE, FEATURE_SIZES, ASPECT_RATIOS, NUM_ANCHORS, batch_first=False, flatten=True)
    test_reader = DataReader(DATA_DIR, IMAGE_SIZE, BATCH_SIZE, FEATURE_SIZES, ASPECT_RATIOS, NUM_ANCHORS, batch_first=False, flatten=False, num_thread=1, queue_size=2)
    print("\nPrepare model")
    model = Model(IMAGE_SIZE + [3], train_reader.num_class, NUM_ANCHORS)
    if model.feature_sizes != FEATURE_SIZES: raise ValueError

    print("\nLoad checkpoint")
    checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model.model)
    checkpoint.restore(tf.train.latest_checkpoint(LOG_DIR))

    summary_writer = tf.summary.create_file_writer(LOG_DIR)

    prev_save_time = time.time()
    data_duration = 0
    train_duration = 0

    print("\nStart iteration\n")
    while True:
        with summary_writer.as_default():
            start_time = time.time()
            images, annotations, classes_gt, locations_gt, default_boxies = train_reader.read_batch()
            data_duration += time.time() - start_time
            
            start_time = time.time()
            loss = model.train(images, classes_gt, locations_gt)
            train_duration += time.time() - start_time

            step = int(model.optimizer.iterations)

            if step % save_freq == 0:
                print(f"step {step}, loss {loss:.3f}, total dur {time.time() - prev_save_time:.2f}, dur {train_duration:.3f} {data_duration:.3f}")
                prev_save_time = time.time()
                data_duration = 0
                train_duration = 0

                checkpoint.save(os.path.join(LOG_DIR, "ckpt"))

                summary_writer.flush()
        
        if step % eval_freq == 0:
            images, annotations, classes_gt, locations_gt, default_boxies = test_reader.read_batch()
            classes_pred, locations_pred = model.predict(images)
            plot_images = draw_prediction(images, classes_pred, locations_pred, default_boxies)


if __name__ == "__main__":
    train()
