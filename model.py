import numpy as np
import tensorflow as tf

#def smooth_l1(x):
#    absx = tf.abs(x)
#    in_threshold = tf.cast((absx < 1.0), tf.float32)
#    over_threshold = 1 - in_threshold
#    x2 = x ** 2
#
#    result = 0.5 * x2 * in_threshold + (absx - 0.5) * over_threshold
#    return result

def smooth_l1(x):
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r

class Model:
    def __init__(self, input_shape, num_class, num_anchors, l=1e-4, lr=1e-3):
        # input_size is tuple of (width, height, channel)
        self.input_shape = input_shape
        self.num_class = num_class
        self.num_anchors = num_anchors

        #self.build_network(l)
        self.build_network_with_batch_normalization(l)

        self.optimizer = tf.optimizers.Adam(lr)

    def build_mobilenet_v1_base_network(self):
        mobilenet_v1 = tf.keras.applications.MobileNet(input_shape=self.input_shape, include_top=False, weights="imagenet")

        base_network = tf.keras.Sequential([l for l in mobilenet_v1.layers[:37]])
        base_network.trainable = False

        base_network.add(tf.keras.layers.Conv2D(512, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(), use_bias=False))
        base_network.add(tf.keras.layers.BatchNormalization())
        base_network.add(tf.keras.layers.ReLU())

        return base_network

    def build_mobilenet_v2_base_network(self):
        mobilenet_v2 = tf.keras.applications.MobileNetV2(input_shape=self.input_shape, include_top=False, weights="imagenet")
        
        base_network = tf.keras.Sequential([l for l in mobilenet_v2.layers[:58]])

    def build_vgg16_base_network(self):
        vgg16 = tf.keras.applications.VGG16(input_shape=self.input_shape, include_top=False, weights="imagenet")

        # 37x37x512
        base_network = tf.keras.Sequential([l for l in vgg16.layers[:14]])
        #self.base_network.trainable = False
        base_network.trainable = True
        for i in range(len(base_network.layers) - 1):
            base_network.layers[i].trainable = False
        
        return base_network

    def build_network(self, l=1e-4):
        feature_maps = []

        # 37x37x512
        base_network = self.build_vgg16_base_network()
        y = base_network(base_network.input)
        feature_maps.append(y)

        # 19x19x1024
        y = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(padding="same"),
            tf.keras.layers.Conv2D(1024, 3, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(1024, 1, padding="same", activation="relu")
        ], name = "output_19")(y)
        feature_maps.append(y)

        # 10x10x512
        y = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(padding="same"),
            tf.keras.layers.Conv2D(256, 1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")
        ], name = "output_10")(y)
        feature_maps.append(y)

        # 5x5x256
        y = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(padding="same"),
            tf.keras.layers.Conv2D(128, 1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")
        ], name = "output_5")(y)
        feature_maps.append(y)

        # 3x3x256
        y = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(padding="same"),
            tf.keras.layers.Conv2D(128, 1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")
        ], name = "output_3")(y)
        feature_maps.append(y)

        # 1x1x256
        y = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(128, 1, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")
        ], name = "output_1")(y)
        feature_maps.append(y)

        self.outputs = []
        for i, f in enumerate(feature_maps):
            output = self.apply_sliding_window(f, self.num_anchors[i])
            self.outputs.append(output)

        self.model = tf.keras.Model(base_network.input, self.outputs)

    def build_network_with_batch_normalization(self, l=1e-4):
        feature_maps = []

        # 37x37x512
        #base_network = self.build_vgg16_base_network()
        base_network = self.build_mobilenet_v1_base_network()
        y = base_network(base_network.input)
        feature_maps.append(y)

        # 19x19x1024
        y = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(padding="same"),
            tf.keras.layers.Conv2D(1024, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(l), use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(1024, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(l), use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU()
        ], name = "output_19")(y)
        feature_maps.append(y)

        # 10x10x512
        y = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(padding="same"),
            tf.keras.layers.Conv2D(256, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(l), use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(512, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(l), use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU()
        ], name = "output_10")(y)
        feature_maps.append(y)

        # 5x5x256
        y = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(padding="same"),
            tf.keras.layers.Conv2D(128, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(l), use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(l), use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU()
        ], name = "output_5")(y)
        feature_maps.append(y)

        # 3x3x256
        y = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(padding="same"),
            tf.keras.layers.Conv2D(128, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(l), use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(l), use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU()
        ], name = "output_3")(y)
        feature_maps.append(y)

        # 1x1x256
        y = tf.keras.Sequential([
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(128, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(l), use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(l), use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.ReLU()
        ], name = "output_1")(y)
        feature_maps.append(y)

        outputs = []
        self.feature_sizes = []
        for i, f in enumerate(feature_maps):
            output = self.apply_sliding_window(f, self.num_anchors[i], l)
            outputs.append(output)
            self.feature_sizes.append(output[0].shape[1])
        
        self.model = tf.keras.Model(base_network.input, outputs)
        # Model output will list of class and location feature map like 
        # [ [[batch, 37, 37, anchor * class], [batch, 37, 37, anchor * 4]], [[batch, 19, 19, anchor * class], [batch, 19, 19, anchor * 4]], [[], []], ... ]

    def apply_sliding_window(self, x, num_anchor, l=1e-4):
        #num_output = (self.num_class + 4) * num_anchor

        #y = tf.keras.layers.Conv2D(num_output, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(l))(x)
        c = tf.keras.layers.Conv2D(num_anchor * self.num_class, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(l))(x)
        l = tf.keras.layers.Conv2D(num_anchor * 4, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(l))(x)
        
        return c, l

    def predict(self, x, training=False):
        classes = []
        locations = []

        predictions = self.model(x, training=training)

        #for i, predict in enumerate(predictions):
        #    p = np.array(predict)
        #    # Reshape to [batch, width, height, anchor, class + location]
        #    p = p.reshape(p.shape[:-1] + (self.num_anchors[i], -1))
        #    c = np.argmax(p[:, :, :, :, :self.num_class], axis=-1)
        #    l = p[:, :, :, :, self.num_class:]
        #
        #    classes.append(c)
        #    locations.append(l)

        for i, o in enumerate(predictions):
            c = o[0]
            l = o[1]
            c = tf.argmax(tf.reshape(c, c.shape[:-1] + (self.num_anchors[i], self.num_class)), axis=-1)
            l = tf.reshape(l, l.shape[:-1] + (self.num_anchors[i], 4))

            classes.append(c)
            locations.append(l)
            
        return classes, locations

    def train(self, images, classes_gt, locations_gt):
        with tf.GradientTape() as tape:
            outputs = self.model(images, training=True)
            
            classes_pred = []
            locations_pred = []

            #for o in outputs:
            #    o = tf.reshape(o, [-1, self.num_class + 4])
            #    classes_pred.append(o[:, :self.num_class])
            #    locations_pred.append(o[:, self.num_class:])

            for c, l in outputs:
                c = tf.reshape(c, [-1, self.num_class])
                l = tf.reshape(l, [-1, 4])
                classes_pred.append(c)
                locations_pred.append(l)
            
            classes_pred = tf.concat(classes_pred, axis=0)
            locations_pred = tf.concat(locations_pred, axis=0)

            loss = self.calculate_loss(classes_pred, locations_pred, classes_gt, locations_gt)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def get_flat_logits_with_tape(self, images, tape):
        # Expect be used for async execution
        with tape:
            outputs = self.model(images, training=True)
            
            classes_pred = []
            locations_pred = []

            #for o in outputs:
            #    o = tf.reshape(o, [-1, self.num_class + 4])
            #    classes_pred.append(o[:, :self.num_class])
            #    locations_pred.append(o[:, self.num_class:])
            
            for c, l in outputs:
                c = tf.reshape(c, [-1, self.num_class])
                l = tf.reshape(l, [-1, 4])
                classes_pred.append(c)
                locations_pred.append(l)
            
            classes_pred = tf.concat(classes_pred, axis=0)
            locations_pred = tf.concat(locations_pred, axis=0)

        return classes_pred, locations_pred

    def calculate_loss_with_tape(self, class_pred, location_pred, class_gt, location_gt, tape, negative_ratio=3.0, step=None):
        # Expect be used for async execution
        with tape:
            loss = self.calculate_loss(class_pred, location_pred, class_gt, location_gt, negative_ratio, step)
        return loss
    
    def calculate_gradients_with_tape(self, loss, tape):
        # Expect be used for async execution
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return gradients

    def apply_gradients(self, gradients):
        # Expect be used for async execution
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def calculate_loss(self, class_pred, location_pred, class_gt, location_gt, negative_ratio=3.0, step=None):
        # Generate hard negative mining mask
        class_softmax = tf.keras.layers.Softmax()(class_pred)
        positive_mask = class_gt != 0
        negative_mask = class_gt == 0

        negative_softmax = -1.0 * class_softmax[:, 0] * negative_mask + -1.0 * positive_mask
        num_negative = max(min(sum(positive_mask) * negative_ratio, sum(negative_mask)), 1)
        _, indices = tf.math.top_k(negative_softmax, k=num_negative)

        negative_mask = np.zeros(class_gt.shape[0]).astype(np.bool)
        for i in indices: negative_mask[i] = True
        mask = positive_mask + negative_mask

        # Class loss
        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(class_gt, class_pred) * mask
        class_loss = tf.reduce_sum(class_loss) / sum(mask)
        
        # Location loss
        delta = location_pred - location_gt
        l1 = smooth_l1(delta)
        location_loss = tf.reduce_sum(tf.reduce_sum(l1, axis=1) * positive_mask) / sum(positive_mask)

        # Regularization loss
        regularization_loss = tf.reduce_sum(self.model.losses)
        
        loss = class_loss + location_loss + regularization_loss

        if step is None: step = self.optimizer.iterations
        tf.summary.scalar("loss", loss, step=step)
        tf.summary.scalar("class_loss", class_loss, step=step)
        tf.summary.scalar("locatin_loss", location_loss, step=step)
        tf.summary.scalar("regularization_loss", regularization_loss, step=step)

        return loss


if __name__ == "__main__":
    import time
    model = Model([300, 300, 3], 21, [4, 6, 6, 6, 4, 4])

    rand = np.random.random([32, 300, 300, 3])
    
    with tf.GradientTape() as tape:
        outputs = model.model(rand)


    c0, l0 = model.predict(rand)

    with tf.GradientTape() as tape:
        start_time = time.time()
        c1, l1 = model.predict(rand)
        print(f"one elapse {time.time() - start_time}s")
    

    print("done")