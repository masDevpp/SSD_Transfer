import os
import numpy as np
import tensorflow as tf
from data_reader import DataReader
from model import Model
import time
from collections import deque
import threading

DATA_DIR = "D:\\MachineLearning\\VOC\\VOCdevkit\\VOC2007"
LOG_DIR = os.path.join(os.getcwd(), "log_async")

save_freq = 10
eval_freq = 10000000

IMAGE_SIZE = [300, 300]
BATCH_SIZE = 8
FEATURE_SIZES = [37, 19, 10, 5, 3, 1]
NUM_ANCHORS = [4, 6, 6, 6, 4, 4]
ASPECT_RATIOS = [1, 2, 1/2, 3, 1/3]

class WorkerContext():
    def __init__(self, model, id):
        self.model = model
        self.id = id

        self.tape = None
        self.classes_pred = None
        self.locations_pred = None
        self.classes_gt = None
        self.locations_gt = None
        self.loss = None
        self.gradients = None
    
    def reset(self, source_variables):
        for target, source in zip(self.model.model.trainable_variables, source_variables):
            target.assign(source)
        
        self.current_time = time.time()
        self.elapse_time_logit = -1
        self.elapse_time_loss = -1
        self.elapse_time_gradient = -1
        
        self.tape = None
        self.classes_pred = None
        self.locations_pred = None
        self.classes_gt = None
        self.locations_gt = None
        self.loss = None
        self.gradients = None


class Main():
    def __init__(self):
        print("\nPrepare reader")
        self.train_reader = DataReader(DATA_DIR, IMAGE_SIZE, BATCH_SIZE, FEATURE_SIZES, ASPECT_RATIOS, NUM_ANCHORS, batch_first=False, flatten=True)
        self.test_reader = DataReader(DATA_DIR, IMAGE_SIZE, BATCH_SIZE, FEATURE_SIZES, ASPECT_RATIOS, NUM_ANCHORS, batch_first=False, flatten=False, num_thread=1, queue_size=2)
        print("\nPrepare model")
        with tf.device("cpu:0"):
            self.model = Model(IMAGE_SIZE + [3], self.train_reader.num_class, NUM_ANCHORS, l=1e-8)
        if self.model.feature_sizes != FEATURE_SIZES: raise ValueError

        print("\nLoad checkpoint")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.model.optimizer, model=self.model.model)
        self.checkpoint.restore(tf.train.latest_checkpoint(LOG_DIR))

        self.train_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, "train"))
        self.test_writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, "test"))

        self.prev_eval_time = time.time()
        self.logit_elapse = 0
        self.loss_elapse = 0
        self.gradient_elapse = 0
        self.logit_wait = 0
        self.loss_wait = 0
        self.gradient_wait = 0
        self.evaluation_wait = 0

        print("\nPrepare events, threads, and queues")
        # Prepare event and thread for logit, loss and gradient calculation
        self.logit_queue = deque([])
        self.logit_event = threading.Event()
        self.logit_event.clear()
        self.logit_thread = threading.Thread(target=self.logit_handler)
        self.logit_thread.start()

        self.num_loss_thread = 2
        self.target_loss_thread = 0
        self.loss_queue = []
        self.loss_event = []
        self.loss_thread = []
        for i in range(self.num_loss_thread):
            self.loss_queue.append(deque([]))
            self.loss_event.append(threading.Event())
            self.loss_event[i].clear()
            self.loss_thread.append(threading.Thread(target=self.loss_handler, args=(i,)))
            self.loss_thread[i].start()

        self.gradient_queue = deque([])
        self.gradient_event = threading.Event()
        self.gradient_event.clear()
        self.gradient_thread = threading.Thread(target=self.gradient_handler)
        self.gradient_thread.start()

        self.evaluation_queue = deque([])
        self.evaluation_event = threading.Event()
        self.evaluation_event.clear()
        self.evaluation_thread = threading.Thread(target=self.evaluation_handler)
        self.evaluation_thread.start()
        
        print("\nPrepare worker contexts")
        # Prepare worker contexts and start calculation
        self.num_worker = 3
        self.worker_contexts = []
        for i in range(self.num_worker):
            model = Model(IMAGE_SIZE + [3], self.train_reader.num_class, NUM_ANCHORS, l=1e-8)
            self.worker_contexts.append(WorkerContext(model, str(i)))
            self.worker_contexts[-1].reset(self.model.model.trainable_variables)

            self.logit_queue.append(self.worker_contexts[-1])
        
        print("\nStart iteration")
        self.logit_event.set()

        self.timer_interval = 120.0
        threading.Timer(self.timer_interval, self.timer_interrupt).start()

    def timer_interrupt(self):
        # Timer interrupt handler to monitor training status
        threading.Timer(self.timer_interval, self.timer_interrupt).start()
    
    def logit_handler(self):
        with tf.device("gpu:0"):
            while True:
                images, annotations, classes_gt, locations_gt, default_boxies = self.train_reader.read_batch()

                # Clear event if queue empty
                if len(self.logit_queue) == 0: self.logit_event.clear()
                
                start_time = time.time()
                self.logit_event.wait()
                self.logit_wait += time.time() - start_time

                worker_context = self.logit_queue.popleft()
                
                # Define GradientTape
                tape = tf.GradientTape()
                worker_context.tape = tape
                
                classes_pred, locations_pred = worker_context.model.get_flat_logits_with_tape(images, worker_context.tape)

                worker_context.classes_pred = classes_pred
                worker_context.locations_pred = locations_pred
                worker_context.classes_gt = classes_gt
                worker_context.locations_gt = locations_gt

                worker_context.elapse_time_logit = time.time() - worker_context.current_time
                worker_context.current_time = time.time()

                self.loss_queue[self.target_loss_thread].append(worker_context)
                self.loss_event[self.target_loss_thread].set()
                self.target_loss_thread += 1
                if self.target_loss_thread == self.num_loss_thread: self.target_loss_thread = 0
    
    def loss_handler(self, index):
        with tf.device("gpu:0"):
            while True:
                if len(self.loss_queue[index]) == 0: self.loss_event[index].clear()
                
                start_time = time.time()
                self.loss_event[index].wait()
                self.loss_wait += time.time() - start_time

                worker_context = self.loss_queue[index].popleft()

                with self.train_writer.as_default():
                    loss = worker_context.model.calculate_loss_with_tape(
                        worker_context.classes_pred, 
                        worker_context.locations_pred, 
                        worker_context.classes_gt,
                        worker_context.locations_gt,
                        worker_context.tape,
                        step=self.model.optimizer.iterations
                        )
                
                worker_context.loss = loss

                worker_context.elapse_time_loss = time.time() - worker_context.current_time
                worker_context.current_time = time.time()

                self.gradient_queue.append(worker_context)
                self.gradient_event.set()

    def gradient_handler(self):
        with tf.device("gpu:0"):
            while True:
                if len(self.gradient_queue) == 0: self.gradient_event.clear()
                
                start_time = time.time()
                self.gradient_event.wait()
                self.gradient_wait += time.time() - start_time

                worker_context = self.gradient_queue.popleft()

                # Calculate gradients using worker model
                gradients = worker_context.model.calculate_gradients_with_tape(worker_context.loss, worker_context.tape)
                worker_context.gradient = gradients

                # Apply gradients to global model
                self.model.apply_gradients(gradients)

                worker_context.elapse_time_gradient = time.time() - worker_context.current_time
                worker_context.current_time = time.time()

                self.evaluation_queue.append(worker_context)
                self.evaluation_event.set()
            
    
    def evaluation_handler(self):
        with tf.device("gpu:0"):
            while True:
                if len(self.evaluation_queue) == 0: self.evaluation_event.clear()
                
                start_time = time.time()
                self.evaluation_event.wait()
                self.evaluation_wait += time.time() - start_time

                worker_context = self.evaluation_queue.popleft()

                step = int(self.model.optimizer.iterations)

                self.logit_elapse += worker_context.elapse_time_logit
                self.loss_elapse += worker_context.elapse_time_loss
                self.gradient_elapse += worker_context.elapse_time_gradient
                
                # Save model or evaluatie performance if necessary
                if step % save_freq == 0:
                    total_elapse = self.logit_elapse + self.loss_elapse + self.gradient_elapse
                    print(f"step {step}, loss {worker_context.loss:.3f}, total dur {time.time() - self.prev_eval_time:.2f}, dur {total_elapse:.2f}({self.logit_elapse:.2f} {self.loss_elapse:.2f} {self.gradient_elapse:.2f}), wait {self.logit_wait:.2f} {self.loss_wait:.2f} {self.gradient_wait:.2f} {self.evaluation_wait:.2f}")
                    self.checkpoint.save(os.path.join(LOG_DIR, "ckpt"))
                    self.train_writer.flush()

                    self.prev_eval_time = time.time()
                    self.logit_elapse = 0
                    self.loss_elapse = 0
                    self.gradient_elapse = 0
                    self.logit_wait = 0
                    self.loss_wait = 0
                    self.gradient_wait = 0
                    self.evaluation_wait = 0

                # Reset and enqueue to logit_queue
                worker_context.reset(self.model.model.trainable_variables)
                self.logit_queue.append(worker_context)
                self.logit_event.set()

            

if __name__ == "__main__":
    main = Main()