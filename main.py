import os
import numpy as np
import tensorflow as tf
from data_reader import DataReader
from model import Model
import time
from PIL import Image, ImageDraw, ImageFont
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
regularizer_coeff = 0.0

def draw_bounding_box0(images, classes_pred, locations_pred, default_boxies, softmaxis, max_num_plot):
    drawn_images = []

    for b in range(min(classes_pred[0].shape[0], max_num_plot)):
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
                        
                        c = classes_pred[f][b, y, x, a]
                        color = f"hsl({(c - 1) / 20 * 360},100%,50%)"
                        font = ImageFont.truetype("arial.ttf", size=15)

                        if softmaxis is None: softmax = ""
                        else: softmax = f"{softmaxis[f][b, y, x, a, c]:.2f}"
                        text = DataReader.class_to_name(None, c) + " " + softmax

                        draw.rectangle([xmin, ymin, xmax, ymax], outline=color)

                        draw.text([xmin + 1, ymin], text, font=font, fill=color)

        drawn_images.append(image)

    return drawn_images

def draw_bounding_box(images, classes_pred, locations_pred, default_boxies, softmaxis, max_num_plot):
    drawn_images = []
    draws = []

    # Image and ImageDraw for image batch
    for image in images:
        image = Image.fromarray((image * 255).astype(np.uint8))
        drawn_images.append(image)
        draws.append(ImageDraw.Draw(image))

    for f in range(len(classes_pred)):
        indicies = np.where(classes_pred[f] != 0)

        for i in range(len(indicies[0])):
            b = indicies[0][i]
            y = indicies[1][i]
            x = indicies[2][i]
            a = indicies[3][i]

            if classes_pred[f][b, y, x, a] == 0: raise ValueError

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
            
            c = classes_pred[f][b, y, x, a]
            color = f"hsl({(c - 1) / 20 * 360},100%,50%)"
            font = ImageFont.truetype("arial.ttf", size=15)

            if softmaxis is None: softmax = ""
            else: softmax = f"{softmaxis[f][b, y, x, a, c]:.2f}"
            text = DataReader.class_to_name(None, c) + " " + softmax

            draw = draws[b]
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color)
            draw.text([xmin + 1, ymin], text, font=font, fill=color)

    return drawn_images

class Main:
    def __init__(self):
        self.drawing = False
        self.train()

    def train(self):
        print("\nPrepare reader")
        train_reader = DataReader(DATA_DIR, IMAGE_SIZE, BATCH_SIZE, FEATURE_SIZES, ASPECT_RATIOS, NUM_ANCHORS, batch_first=False, flatten=True, num_thread=6, queue_size=8)
        test_reader = DataReader(TEST_DATA_DIR, IMAGE_SIZE, TEST_BATCH_SIZE, FEATURE_SIZES, ASPECT_RATIOS, NUM_ANCHORS, batch_first=False, flatten=True, distort_prob=0.2)
        train_reader_unflat = DataReader(DATA_DIR, IMAGE_SIZE, BATCH_SIZE, FEATURE_SIZES, ASPECT_RATIOS, NUM_ANCHORS, batch_first=False, flatten=False)
        test_reader_unflat = DataReader(TEST_DATA_DIR, IMAGE_SIZE, TEST_BATCH_SIZE, FEATURE_SIZES, ASPECT_RATIOS, NUM_ANCHORS, batch_first=False, flatten=False, distort_prob=0.2)

        images, annotations, classes_gt, locations_gt, default_boxies = test_reader_unflat.read_batch()
        stt = time.time()
        drawn_images1 = draw_bounding_box(images, classes_gt, locations_gt, default_boxies, None, 100)
        print(f"1 {time.time() - stt:.3f}")
        stt = time.time()
        drawn_images2 = draw_bounding_box2(images, classes_gt, locations_gt, default_boxies, None, 100)
        print(f"2 {time.time() - stt:.3f}")
        
        print("\nPrepare model")
        learnining_rate = tf.optimizers.schedules.ExponentialDecay(0.01, train_reader.num_data / BATCH_SIZE, 0.95, staircase=False)
        model = Model(IMAGE_SIZE + [3], train_reader.num_class, NUM_ANCHORS, regularizer_coeff=regularizer_coeff, lr=learnining_rate)
        if model.feature_sizes != FEATURE_SIZES: raise ValueError

        print("\nLoad checkpoint")
        checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model.model)
        cp_manager = tf.train.CheckpointManager(checkpoint, LOG_DIR, 3, keep_checkpoint_every_n_hours=4)
        checkpoint.restore(cp_manager.latest_checkpoint)

        train_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, "train"))
        test_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, "test"))

        prev_save_time = time.time()
        data_duration = 0
        train_duration = 0

        print("\nStart iteration\n")
        while True:
            with train_writer.as_default():
                start_time = time.time()
                images, annotations, classes_gt, locations_gt, default_boxies = train_reader.read_batch()
                data_duration += time.time() - start_time
                
                start_time = time.time()
                loss = model.train(images, classes_gt, locations_gt)
                train_duration += time.time() - start_time

                step = int(model.optimizer.iterations)
                
                if step % print_freq == 0:
                    epoch = step * BATCH_SIZE / train_reader.num_data
                    print(f"step {step}, {epoch:.2f} epoch, loss {loss:.3f}, total dur {time.time() - prev_save_time:.2f}, dur {train_duration:.3f} {data_duration:.3f}, queue {len(train_reader.batch_queue)}/{train_reader.queue_size}, {time.asctime()}")
                    prev_save_time = time.time()
                    data_duration = 0
                    train_duration = 0
                    
                if step % save_freq == 0:
                    #checkpoint.save(os.path.join(LOG_DIR, "ckpt_" + str(step)))
                    cp_manager.save()
                    tf.summary.scalar("learning_rate", model.optimizer.lr(model.optimizer.iterations), step=model.optimizer.iterations)
                    train_writer.flush()
            
            if step % eval_freq == 0:
                # Eval train data
                images, annotations, classes_gt, locations_gt, default_boxies = train_reader_unflat.read_batch()
                classes_pred, locations_pred, softmaxis = model.predict(images, training=True)
                threading.Thread(target=self.draw, args=(images, classes_pred, locations_pred, default_boxies, softmaxis, TEST_BATCH_SIZE, model.optimizer.iterations, train_writer), name="train_draw").start()

                # Eval test data
                images, annotations, classes_gt, locations_gt, default_boxies = test_reader.read_batch()
                with test_writer.as_default():
                    classes_pred, locations_pred = model.get_flat_logits(images)
                    loss = model.calculate_loss(classes_pred, locations_pred, classes_gt, locations_gt)

                images, annotations, classes_gt, locations_gt, default_boxies = test_reader_unflat.read_batch()
                classes_pred, locations_pred, softmaxis = model.predict(images, training=True)
                threading.Thread(target=self.draw, args=(images, classes_pred, locations_pred, default_boxies, softmaxis, TEST_BATCH_SIZE, model.optimizer.iterations, test_writer), name="test_draw").start()

    def draw(self, images, classes_pred, locations_pred, default_boxies, softmaxis, max_num_plot, step, summary_writer):
        drawn_images = draw_bounding_box(images, classes_pred, locations_pred, default_boxies, softmaxis, max_num_plot)

        image_array = [np.array(image) for image in drawn_images]
        image_array = np.array(image_array).astype(np.uint8)

        with summary_writer.as_default():
            tf.summary.image("predict", image_array, step=step)
        summary_writer.flush()


if __name__ == "__main__":
    main = Main()
