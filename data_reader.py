import os
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import glob
import xml.etree.ElementTree as et
from collections import deque
import threading
import time

class DataReader:
    def __init__(self, base_dir, image_size, batch_size, feature_sizes, aspect_ratios, num_anchors, batch_first=False, flatten=True, distort_prob=0.5, num_thread=2, queue_size=4):
        self.debug_output = False
        self.num_class = 21
        
        self.base_dir = base_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.feature_sizes = feature_sizes
        self.aspect_ratios = aspect_ratios
        self.num_anchors = num_anchors
        self.batch_first = batch_first
        self.flatten = flatten
        self.distort_prob = distort_prob

        self.batch_queue = deque([])
        self.queue_size = queue_size

        # Get annotation file names without ext
        self.file_names = []
        for path in glob.iglob(os.path.join(self.base_dir, "Annotations", "*.xml")):
            self.file_names.append(os.path.splitext(os.path.basename(path))[0])
        self.num_data = len(self.file_names)

        # Handle thread after prepare all
        self.event = threading.Event()
        self.event.clear()

        self.threads = []
        for i in range(num_thread):
            self.threads.append(threading.Thread(target=self.read_batch_worker, daemon=True, name="data_reader"))
            self.threads[i].start()

        self.event.set()
    
    def read_batch(self):
        if self.debug_output and len(self.batch_queue) == 0: print("queue empty!")
        while len(self.batch_queue) == 0: continue

        if self.debug_output: print("dequeue")
        batch = self.batch_queue.popleft()
        
        if len(self.batch_queue) < self.queue_size:
            self.event.set()

        return batch
        
    def read_data(self):
        index = np.random.randint(len(self.file_names))
        image_file = os.path.join(self.base_dir, "JPEGImages", self.file_names[index] + ".jpg")
        annotation_file = os.path.join(self.base_dir, "Annotations", self.file_names[index] + ".xml")

        # Load image
        image = Image.open(image_file)

        # Load annotation data
        class AnnotationData:
            pass

        annotation_datas = []

        root = et.parse(annotation_file).getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            xmax = float(bndbox.find("xmax").text)
            ymin = float(bndbox.find("ymin").text)
            ymax = float(bndbox.find("ymax").text)
            width = xmax - xmin
            height = ymax - ymin
            xcenter = xmin + width / 2
            ycenter = ymin + height / 2

            ad = AnnotationData()
            ad.file_name = self.file_names[index]
            ad.name = name
            ad.c = self.name_to_class(ad.name)
            ad.xmin = xmin
            ad.xmax = xmax
            ad.ymin = ymin
            ad.ymax = ymax
            ad.width = width
            ad.height = height
            ad.xcenter = xcenter
            ad.ycenter = ycenter

            annotation_datas.append(ad)
        
        return image, annotation_datas
    
    def distort_image(self, image, distort_min_ratio=0.6, distort_max_ratio=1.4):
        # Brightness
        enhancer = ImageEnhance.Brightness(image)
        ratio = np.random.random() * (distort_max_ratio - distort_min_ratio) + distort_min_ratio
        image = enhancer.enhance(ratio)
        
        # Contrast
        enhancer = ImageEnhance.Contrast(image)
        ratio = np.random.random() * (distort_max_ratio - distort_min_ratio) + distort_min_ratio
        image = enhancer.enhance(ratio)

        return image

    def flip_image(self, image, annotations, flip_ratio):
        if np.random.random() < flip_ratio:
            image = image.transpose(method=Image.FLIP_LEFT_RIGHT)

            for annotation in annotations:
                annotation.xcenter = annotation.width - annotation.xcenter
                
                orig_xmin = annotation.xmin
                orig_xmax = annotation.xmax
                annotation.xmin = image.width - orig_xmax
                annotation.xmax = image.width - orig_xmin
        
        return image, annotations

    def crop_image(self, image, annotations, size, size_variance, size_ratio=None, size_variance_ratio=None):
        if size_ratio is not None:
            # Overwrite size using ratio
            size[0] = image.size[0] * size_ratio
            size[1] = image.size[1] * size_ratio        
        
        if size_variance_ratio is not None:
            # Overwrite size_variance using ratio
            size_variance[0] = image.size[0] * size_variance_ratio
            size_variance[1] = image.size[1] * size_variance_ratio

        width = int(size[0] + (np.random.random() - 0.5) * 2 * size_variance[0])
        height = int(size[1] + (np.random.random() - 0.5) * 2 * size_variance[1])
        
        if width > image.size[0]: 
            width = image.size[0]
            xmin = 0
        else:
            xmin = np.random.randint(0, image.size[0] - width)

        if height > image.size[1]: 
            height = image.size[1]
            ymin = 0
        else:
            ymin = np.random.randint(0, image.size[1] - height)
        
        xmax = xmin + width
        ymax = ymin + height

        new_annotations = []
        for annotation in annotations:
            # Skip if out of crop extent
            if annotation.xmin >= xmax or annotation.xmax <= xmin or annotation.ymin >= ymax or annotation.ymax <= ymin: continue

            # Cap annotation extent
            if annotation.xmin < xmin: annotation.xmin = xmin
            if annotation.xmax > xmax: annotation.xmax = xmax
            if annotation.ymin < ymin: annotation.ymin = ymin
            if annotation.ymax > ymax: annotation.ymax = ymax

            # Shift
            annotation.xmin -= xmin
            annotation.xmax -= xmin
            annotation.ymin -= ymin
            annotation.ymax -= ymin
            annotation.width = annotation.xmax - annotation.xmin
            annotation.height = annotation.ymax - annotation.ymin
            annotation.xcenter = annotation.xmin + annotation.width / 2
            annotation.ycenter = annotation.ymin + annotation.height / 2

            new_annotations.append(annotation)
        
        image = image.crop((xmin, ymin, xmax, ymax))

        return image, new_annotations

    def resize_image(self, image, annotations, size, size_variance):
        width = int(size[0] + (np.random.random() - 0.5) * 2 * size_variance)
        height = int(size[1] + (np.random.random() - 0.5) * 2 * size_variance)

        width_ratio = width / image.size[0]
        height_ratio = height / image.size[1]

        for annotation in annotations:
            annotation.xmin *= width_ratio
            annotation.xmax *= width_ratio
            annotation.ymin *= height_ratio
            annotation.ymax *= height_ratio

            annotation.width *= width_ratio
            annotation.height *= height_ratio

            annotation.xcenter *= width_ratio
            annotation.ycenter *= height_ratio
        
        image = image.resize((width, height), resample=Image.BILINEAR)

        return image, annotations
    
    def normalize_annotation(self, image, annotations):
        for annotation in annotations:
            annotation.xmin /= image.width
            annotation.xmax /= image.width
            annotation.ymin /= image.height
            annotation.ymax /= image.height

            annotation.width /= image.width
            annotation.height /= image.height

            annotation.xcenter /= image.width
            annotation.ycenter /= image.height

        return annotations

    def read_batch_worker(self):
        while True:
            images = []
            annotations = []
            classes_gt = []
            locations_gt = []
            default_boxies = []

            for _ in range(self.batch_size):
                image, annotation = self.read_data()

                # Image distortion
                if np.random.random() < self.distort_prob:
                    image, annotation = self.flip_image(image, annotation, 1.0)
                    image, annotation = self.crop_image(image, annotation, [0, 0], [0, 0], 0.8, 0.2)
                    image = self.distort_image(image)
                
                image, annotation = self.resize_image(image, annotation, self.image_size, 0)
                annotation = self.normalize_annotation(image, annotation)

                # Normalize image
                image = np.array(image) / 255

                images.append(image)
                annotations.append(annotation)

                cl = []
                loc = []
                db = []
                for i, fs in enumerate(self.feature_sizes):
                    c, l, d = self.generate_ground_truth(annotation, fs, self.aspect_ratios[:self.num_anchors[i] - 1])
                    cl.append(c)
                    loc.append(l)
                    db.append(d)
                classes_gt.append(cl)
                locations_gt.append(loc)
                default_boxies.append(db)

            if self.batch_first == False:
                # Reshape to [feature, batch, height, width, anchor, class or location]
                cl = []
                loc = []
                db = []
                for f in range(len(classes_gt[0])):
                    c = []
                    l = []
                    d = []
                    for b in range(len(classes_gt)):
                        c.append(classes_gt[b][f])
                        l.append(locations_gt[b][f])
                        d.append(default_boxies[b][f])
                    c = np.array(c)
                    l = np.array(l)
                    d = np.array(d)

                    if self.flatten:
                        c = c.reshape([-1])
                        l = l.reshape([-1, l.shape[-1]])
                        d = d.reshape([-1, d.shape[-1]])

                    cl.append(np.array(c))
                    loc.append(np.array(l))
                    db.append(np.array(d))
                classes_gt = cl
                locations_gt = loc
                default_boxies = db

                if self.flatten:
                    # Reshape to [feature * batch * height * width * anchor, class or location]
                    classes_gt = np.concatenate(classes_gt, axis=0)
                    locations_gt = np.concatenate(locations_gt, axis=0)
                    default_boxies = np.concatenate(default_boxies, axis=0)

            batch = [np.array(images), annotations, classes_gt, locations_gt, default_boxies]

            if len(self.batch_queue) >= self.queue_size:
                self.event.clear()

            self.event.wait()
            if self.debug_output: print("euqueue: " + str(len(self.batch_queue)) + ", " + str(threading.get_ident()))
            self.batch_queue.append(batch)
    
    def name_to_class(self, name):
        if name == "aeroplane": c = 1
        elif name == "bicycle": c = 2
        elif name == "bird": c = 3
        elif name == "boat": c = 4
        elif name == "bottle": c = 5
        elif name == "bus": c = 6
        elif name == "car": c = 7
        elif name == "cat": c = 8
        elif name == "chair": c = 9
        elif name == "cow": c = 10
        elif name == "diningtable": c = 11
        elif name == "dog": c = 12
        elif name == "horse": c = 13
        elif name == "motorbike": c = 14
        elif name == "person": c = 15
        elif name == "pottedplant": c = 16
        elif name == "sheep": c = 17
        elif name == "sofa": c = 18
        elif name == "train": c = 19
        elif name == "tvmonitor": c = 20
        else:
            ValueError("Invalid name")
        return c

    def class_to_name(self, c):
        if c == 1: name = "aeroplane"
        elif c == 2: name = "bicycle"
        elif c == 3: name = "bird"
        elif c == 4: name = "boat"
        elif c == 5: name = "bottle"
        elif c == 6: name = "bus"
        elif c == 7: name = "car"
        elif c == 8: name = "cat"
        elif c == 9: name = "chair"
        elif c == 10: name = "cow"
        elif c == 11: name = "diningtable"
        elif c == 12: name = "dog"
        elif c == 13: name = "horse"
        elif c == 14: name = "motorbike"
        elif c == 15: name = "person"
        elif c == 16: name = "pottedplant"
        elif c == 17: name = "sheep"
        elif c == 18: name = "sofa"
        elif c == 19: name = "train"
        elif c == 20: name = "tvmonitor"
        else:
            ValueError("Invalid name")
        return name

    def generate_ground_truth(self, annotations, feature_size, aspect_ratio_list, overlap_threshold=0.5, sk=1, sk2=1):
        classies = np.zeros([feature_size, feature_size, len(aspect_ratio_list) + 1]).astype(np.int32)
        locations = np.zeros([feature_size, feature_size, len(aspect_ratio_list) + 1, 4])
        default_boxies = np.zeros([feature_size, feature_size, len(aspect_ratio_list) + 1, 4])

        overlap_max_max = 0

        for x in range(feature_size):
            for y in range(feature_size):
                cx = (x + 0.5) / feature_size
                cy = (y + 0.5) / feature_size

                for aspect_index in range(len(aspect_ratio_list) + 1):
                    # Add one to len(aspect_ratio_list) to handle additional larger square default box
                    if aspect_index == len(aspect_ratio_list):
                        #width = np.sqrt(sk * sk2)
                        #height = np.sqrt(sk * sk2)
                        width = 2.8
                        height = 2.8
                    else:
                        #width = sk * np.sqrt(aspect_ratio_list[aspect_index])
                        #height = sk / np.sqrt(aspect_ratio_list[aspect_index])
                        width = np.sqrt(aspect_ratio_list[aspect_index])
                        height = 1 / np.sqrt(aspect_ratio_list[aspect_index])
                    
                    width = width / feature_size
                    height = height / feature_size

                    xmin = cx - width / 2
                    xmax = cx + width / 2
                    ymin = cy - height / 2
                    ymax = cy + height / 2

                    default_boxies[y, x, aspect_index, 0] = cx
                    default_boxies[y, x, aspect_index, 1] = cy
                    default_boxies[y, x, aspect_index, 2] = width
                    default_boxies[y, x, aspect_index, 3] = height

                    overlap_max = 0
                    for annotation in annotations:
                        # Skip if center is not in annotation bounding box
                        if xmin > annotation.xmax or xmax < annotation.xmin or ymin > annotation.ymax or ymax < annotation.ymin: continue

                        overlap_xmin = max(xmin, annotation.xmin)
                        overlap_xmax = min(xmax, annotation.xmax)
                        overlap_ymin = max(ymin, annotation.ymin)
                        overlap_ymax = min(ymax, annotation.ymax)

                        overlap = (overlap_xmax - overlap_xmin) * (overlap_ymax - overlap_ymin)
                        overlap = overlap / (width * height + annotation.width * annotation.height - overlap)

                        # overlap_max to find the largets overlap object
                        if overlap >= overlap_threshold and overlap > overlap_max:
                            overlap_max = overlap

                            if overlap > overlap_max_max: overlap_max_max = overlap

                            classies[y, x, aspect_index] = annotation.c
                            locations[y, x, aspect_index, 0] = (annotation.xcenter - cx) / width
                            locations[y, x, aspect_index, 1] = (annotation.ycenter - cy) / height
                            locations[y, x, aspect_index, 2] = np.log(annotation.width / width)
                            locations[y, x, aspect_index, 3] = np.log(annotation.height / height)

        locations = locations.astype(np.float32)
        default_boxies = default_boxies.astype(np.float32)
        
        #print(f"overlap_max_max: {overlap_max_max}")
        return classies, locations, default_boxies


if __name__ == "__main__":
    width = 300
    height = 300
    aspect_ratio_list = [1, 2, 1/2, 3, 1/3]
    #aspect_ratio_list = [1, 2, 3, 5, 7, 9, 12, 24, 1/2, 1/3, 1/5, 1/7, 1/9, 1/12, 1/24]
    #feature_size = [37, 18, 9, 4, 2, 1]
    feature_size = [37, 19, 10, 5, 3, 1]
    num_anchors = [4, 6, 6, 6, 4, 4]
    reader = DataReader("D:\\MachineLearning\\VOC\\VOCdevkit\\VOC2007", (width, height), 8, feature_size, aspect_ratio_list, num_anchors, num_thread=1)

    images, annotations, classes_gt, locations_gt, default_boxies_gt = reader.read_batch()
    for image, annotation, classes, locations, default_boxies in zip(images, annotations, classes_gt, locations_gt, default_boxies_gt):
        image_gt = image

        image_gt = Image.fromarray((image_gt * 255).astype(np.uint8))
        draw_gt = ImageDraw.Draw(image_gt)
        for anno_gt in annotation:
            draw_gt.rectangle([anno_gt.xmin * width, anno_gt.ymin * height, anno_gt.xmax * width, anno_gt.ymax * height], outline="red")
            print(anno_gt.name)

        image_gt.show()

        image = Image.fromarray((image * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)

        smin = 0.2
        smax = 0.9

        for i, fs in enumerate(feature_size):
            sk = smin + (smax - smin) / (len(feature_size) - 1) * i
            sk2 = smin + (smax - smin) / (len(feature_size) - 1) * (i + 1)

            c, l, d = reader.generate_ground_truth(annotation, fs, aspect_ratio_list, overlap_threshold=0.4)
            #c = classes[i]
            #l = locations[i]
            #d = default_boxies[i]

            for y in range(c.shape[0]):
                for x in range(c.shape[1]):
                    for asp in range(c.shape[2]):
                        if c[y, x, asp] == 0: continue

                        xmin = (d[y, x, asp, 0] - d[y, x, asp, 2] / 2) * width
                        xmax = (d[y, x, asp, 0] + d[y, x, asp, 2] / 2) * width
                        ymin = (d[y, x, asp, 1] - d[y, x, asp, 3] / 2) * height
                        ymax = (d[y, x, asp, 1] + d[y, x, asp, 3] / 2) * height
                        #draw.rectangle([xmin, ymin, xmax, ymax], outline="blue")
                        
                        cx = (l[y, x, asp, 0] * d[y, x, asp, 2] + d[y, x, asp, 0]) * width
                        cy = (l[y, x, asp, 1] * d[y, x, asp, 3] + d[y, x, asp, 1]) * height
                        wid = (np.exp(1) ** l[y, x, asp, 2]) * d[y, x, asp, 2] * width
                        hei = (np.exp(1) ** l[y, x, asp, 3]) * d[y, x, asp, 3] * height
                        draw.rectangle([cx - wid / 2, cy - hei / 2, cx + wid / 2, cy + hei / 2], outline="red")
                        

        image.show()

    for i in range(1000):
        out = reader.read_batch()

    print("done")