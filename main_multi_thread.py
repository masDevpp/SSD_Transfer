import os
import numpy as np
import tensorflow as tf
from data_reader import DataReader
from model import Model
import time
from PIL import Image, ImageDraw
import threading

DATA_DIR = "D:\\MachineLearning\\VOC\\VOCdevkit\\VOC2007"
LOG_DIR = "D:\\MachineLearning\\SSD_Transfer\\log_multi"

save_freq = 100
save_freq_local = save_freq / 5
test_freq = save_freq * 5
eval_freq = 10000000

IMAGE_SIZE = [300, 300]
BATCH_SIZE = 8#16
TEST_BATCH_SIZE = int(BATCH_SIZE / 2)
FEATURE_SIZES = [37, 19, 10, 5, 3, 1]
NUM_ANCHORS = [4, 6, 6, 6, 4, 4]
ASPECT_RATIOS = [1, 2, 1/2, 3, 1/3]
regularizer_coeff = 1e-8

class Main:
    def __init__(self):
        print("\nPrepare reader")
        self.train_reader = DataReader(DATA_DIR, IMAGE_SIZE, BATCH_SIZE, FEATURE_SIZES, ASPECT_RATIOS, NUM_ANCHORS, batch_first=False, flatten=True)
        self.test_reader = DataReader(DATA_DIR, IMAGE_SIZE, TEST_BATCH_SIZE, FEATURE_SIZES, ASPECT_RATIOS, NUM_ANCHORS, batch_first=False, flatten=False, num_thread=1, queue_size=2)
        print("\nPrepare model")
        self.global_model = Model(IMAGE_SIZE + [3], self.train_reader.num_class, NUM_ANCHORS, l=regularizer_coeff)
        if self.global_model.feature_sizes != FEATURE_SIZES: raise ValueError

        print("\nLoad checkpoint")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.global_model.optimizer, model=self.global_model.model)
        self.checkpoint.restore(tf.train.latest_checkpoint(LOG_DIR))

        self.saving = False
        self.plotting = False
        self.previous_time = time.time()
        self.previous_step = int(self.global_model.optimizer.iterations)
        self.previous_test_step = self.previous_step
        self.data_duration = 0
        self.train_duration = 0

        print("\nStart threads")
        self.threads = []
        for i in range(2):
            model = Model(IMAGE_SIZE + [3], self.train_reader.num_class, NUM_ANCHORS, l=regularizer_coeff)
            if model.feature_sizes != FEATURE_SIZES: raise ValueError

            self.threads.append(threading.Thread(target=self.trainer, args=(str(i), self.train_reader, model), name="trainer"))
            self.threads[i].start()

    def callback(self, id, loss, data_duration, train_duration):
        step = int(self.global_model.optimizer.iterations)
        self.data_duration += data_duration
        self.train_duration += train_duration

        if self.saving == False and step - self.previous_step > save_freq:
            self.saving = True

            print(f"step {step}, loss {loss:.3f}, total dur {time.time() - self.previous_time:.2f}, {self.train_duration:.2f} {self.data_duration:.2f}, {time.asctime()}")
            self.checkpoint.save(os.path.join(LOG_DIR, "ckpt"))

            if step - self.previous_test_step > test_freq:
                #self.evaluation()
                threading.Thread(target=self.evaluation, name="evaluation").start()
                self.previous_test_step = step

            self.previous_time = time.time()
            self.previous_step = step
            self.data_duration = 0
            self.train_duration = 0

            self.saving = False

    def evaluation(self):
        if self.plotting: return
        self.plotting = True

        summary_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, "test"))

        with summary_writer.as_default():
            images, annotations, classes_gt, locations_gt, default_boxies = self.test_reader.read_batch()
            classes_pred, locations_pred = self.global_model.predict(images)
            #loss = self.global_model.calculate_loss(classes_pred, locations_pred, classes_gt, locations_gt)

            plot_images = self.draw_prediction(images, classes_pred, locations_pred, default_boxies)
            image_array = [np.array(image) for image in plot_images]
            image_array = np.array(image_array).astype(np.uint8)
            tf.summary.image("test", np.array(image_array), self.global_model.optimizer.iterations)
        
        summary_writer.flush()
        self.plotting = False

        # Plot takes long time so isolate thread
        #thread = threading.Thread(target=self.plot, args=(images, classes_pred, locations_pred, default_boxies, self.global_model.optimizer.iterations))
        #thread.start()
        
    def plot(self, images, classes_pred, locations_pred, default_boxies, step):
        if self.plotting: return
        self.plotting = True
        summary_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, "test"))

        with summary_writer.as_default():
            plot_images = self.draw_prediction(images, classes_pred, locations_pred, default_boxies)
            image_array = [np.array(image) for image in plot_images]
            tf.summary.image("test", image_array, step)
        
        summary_writer.flush()
        self.plotting = False

    def trainer(self, id, train_reader, model):
        summary_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, id))

        itr = 0
        data_duration = 0
        train_duration = 0
        while True:
            # Update variables
            for target, source in zip(model.model.trainable_variables, self.global_model.model.trainable_variables):
                target.assign(source)

            # Compute gradients
            with summary_writer.as_default():
                start_time = time.time()
                images, _, classes_gt, locations_gt, _ = train_reader.read_batch()
                data_duration += time.time() - start_time
                
                start_time = time.time()

                tape = tf.GradientTape()
                classes_pred, locations_pred = model.get_flat_logits_with_tape(images, tape)
                loss = model.calculate_loss_with_tape(classes_pred, locations_pred, classes_gt, locations_gt, tape, step=self.global_model.optimizer.iterations)
                gradients = model.calculate_gradients_with_tape(loss, tape)

                train_duration += time.time() - start_time

                self.global_model.apply_gradients(gradients)
            
            if itr % save_freq_local == 0:
                self.callback(id, loss, data_duration, train_duration)
                data_duration = 0
                train_duration = 0

            itr += 1


    def draw_prediction(self, images, classes_pred, locations_pred, default_boxies):
        drawn_images = []

        for b in range(classes_pred[0].shape[0]):
            image = Image.fromarray((images[b] * 255).astype(np.uint8))
            draw = ImageDraw.Draw(image)

            for f in range(len(classes_pred)):
                for x in range(classes_pred[f].shape[1]):
                    for y in range(classes_pred[f].shape[2]):
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

if __name__ == "__main__":
    main = Main()
