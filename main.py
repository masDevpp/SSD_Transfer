import os
import numpy as np
import tensorflow as tf
from data_reader import DataReader
from model import Model
import time
from PIL import Image, ImageDraw
import threading

DATA_DIR = "D:\\MachineLearning\\VOC\\VOCdevkit\\VOC2007"
TEST_DATA_DIR = "D:\\MachineLearning\\VOC\\2012\\VOCdevkit\\VOC2012"
LOG_DIR = os.path.join(os.getcwd(), "log")

print_freq = 100
save_freq = 500
eval_freq = save_freq * 4

IMAGE_SIZE = [300, 300]
BATCH_SIZE = 8#16
FEATURE_SIZES = [37, 19, 10, 5, 3, 1]
NUM_ANCHORS = [4, 6, 6, 6, 4, 4]
TEST_BATCH_SIZE = BATCH_SIZE // 2
ASPECT_RATIOS = [1, 2, 1/2, 3, 1/3]
regularizer_coeff = 1e-12

def draw_bounding_box(images, classes_pred, locations_pred, default_boxies):
    drawn_images = []

    for b in range(classes_pred[0].shape[0]):
        image = Image.fromarray((images[b] * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)

        for f in range(len(classes_pred)):
            for y in range(classes_pred[f].shape[1]):
                for x in range(classes_pred[f].shape[2]):
                    for a in range(classes_pred[f].shape[3]):
                        if classes_pred[f][b, y, x, a] == 0: continue

                        cx = (locations_pred[f][b, y, x, a, 0] * default_boxies[f][b, y, x, a, 2] + default_boxies[f][b, y, x, a, 0]) * image.size[0]
                        cy = (locations_pred[f][b, y, x, a, 1] * default_boxies[f][b, y, x, a, 3] + default_boxies[f][b, y, x, a, 1]) * image.size[1]
                        wid = (np.exp(1) ** locations_pred[f][b, y, x, a, 2]) * default_boxies[f][b, y, x, a, 2] * image.size[0]
                        hei = (np.exp(1) ** locations_pred[f][b, y, x, a, 3]) * default_boxies[f][b, y, x, a, 3] * image.size[1]

                        xmin = cx - wid / 2
                        xmax = cx + wid / 2
                        ymin = cy - hei / 2
                        ymax = cy + hei / 2
                        if xmin < 0: xmin = 0
                        if xmax > image.size[0] - 1: xmax = image.size[0] - 1
                        if ymin < 0: ymin = 0
                        if ymax > image.size[1] - 1: ymax = image.size[0] - 1

                        draw.rectangle([xmin, ymin, xmax, ymax], outline="red")
                        draw.text([xmin, ymin], DataReader.class_to_name(None, classes_pred[f][b, y, x, a]), fill="red")

        drawn_images.append(image)

    return drawn_images

class Main:
    def __init__(self):
        self.drawing = False
        self.train()

    def train(self):
        print("\nPrepare reader")
        train_reader = DataReader(DATA_DIR, IMAGE_SIZE, BATCH_SIZE, FEATURE_SIZES, ASPECT_RATIOS, NUM_ANCHORS, batch_first=False, flatten=True, num_thread=6, queue_size=8)
        test_reader = DataReader(TEST_DATA_DIR, IMAGE_SIZE, TEST_BATCH_SIZE, FEATURE_SIZES, ASPECT_RATIOS, NUM_ANCHORS, batch_first=False, flatten=False, num_thread=1, queue_size=2, distort_prob=0.2)
        print("\nPrepare model")
        learnining_rate = tf.optimizers.schedules.ExponentialDecay(0.01, train_reader.num_data / BATCH_SIZE, 0.95, staircase=False)
        model = Model(IMAGE_SIZE + [3], train_reader.num_class, NUM_ANCHORS, regularizer_coeff=regularizer_coeff, lr=learnining_rate)
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
                
                if step % print_freq == 0:
                    epoch = step * BATCH_SIZE / train_reader.num_data
                    print(f"\rstep {step}, {epoch:.2f} epoch, loss {loss:.3f}, total dur {time.time() - prev_save_time:.2f}, dur {train_duration:.3f} {data_duration:.3f}, {time.asctime()}")
                    prev_save_time = time.time()
                    data_duration = 0
                    train_duration = 0
                    tf.summary.scalar("learning_rate", model.optimizer.lr(model.optimizer.iterations), step=model.optimizer.iterations)

                if step % save_freq == 0:
                    checkpoint.save(os.path.join(LOG_DIR, "ckpt_" + str(step)))
                    summary_writer.flush()
            
            if step % eval_freq == 0:
                images, annotations, classes_gt, locations_gt, default_boxies = test_reader.read_batch()
                classes_pred, locations_pred = model.predict(images, training=True)
                threading.Thread(target=self.draw, args=(images, classes_pred, locations_pred, default_boxies, model.optimizer.iterations), name="draw").start()
                #plot_images = draw_bounding_box(images, classes_pred, locations_pred, default_boxies)

    def draw(self, images, classes_pred, locations_pred, default_boxies, step):
        if self.drawing: return
        self.drawing = True

        drawn_images = draw_bounding_box(images, classes_pred, locations_pred, default_boxies)

        image_array = [np.array(image) for image in drawn_images]
        image_array = np.array(image_array).astype(np.uint8)

        test_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, "test"))
        
        with test_writer.as_default():
            tf.summary.image("predict", image_array, step=step)
        test_writer.flush()

        self.drawing = False

if __name__ == "__main__":
    main = Main()
