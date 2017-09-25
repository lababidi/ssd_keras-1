import argparse
import os
from math import ceil

import numpy as np
import pandas
import rasterio
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam

from batch_gen import BatchGenerator
from keras_ssd300 import ssd_300
from keras_ssd_loss import SSDLoss
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y2

from keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument('labels', help="labels csv file ['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id']")
parser.add_argument('--test', help="test labels csv file")
parser.add_argument('--batch_size', default=4)
parser.add_argument('--epochs', default=1000)
parser.add_argument('--classes', help="list of integers of classes to include",
                    default=None, type=lambda s: [int(cl) for cl in s.split(',')])
parser.add_argument('--var_dir', help="Validation directory with images")
parser.add_argument('--model', help="model name to be used for weights file", default='ssd_300')
args = parser.parse_args()


img_height = 300  # Height of the input images
img_width = 300  # Width of the input images
img_channels = 3  # Number of color channels of the input images
n_classes = len(args.classes)+1 if args.classes else 2  # Number of classes including the background class, e.g. 21 for the Pascal VOC datasets
scales = [0.2] * 7
aspect_ratios = [[1.0]] * 6
two_boxes_for_ar1 = True
limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2]
# The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'minmax'
# Whether the box coordinates to be used as targets for the model should be in the 'centroids' or 'minmax' format,
normalize_coords = False


K.clear_session()
model, predictor_sizes = ssd_300(image_size=(img_height, img_width, img_channels),
                                 n_classes=n_classes,
                                 min_scale=None,
                                 max_scale=None,
                                 scales=scales,
                                 aspect_ratios_global=None,
                                 aspect_ratios_per_layer=aspect_ratios,
                                 two_boxes_for_ar1=two_boxes_for_ar1,
                                 limit_boxes=limit_boxes,
                                 variances=variances,
                                 coords=coords,
                                 normalize_coords=normalize_coords)
model.load_weights('./ssd300_weights.h5', by_name=True)

model.compile(
    optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05),
    loss=SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=0.1).compute_loss)

ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=None,
                                max_scale=None,
                                scales=scales,
                                aspect_ratios_global=None,
                                aspect_ratios_per_layer=aspect_ratios,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.4,
                                neg_iou_threshold=0.2,
                                coords=coords,
                                normalize_coords=normalize_coords)

labels_format = ['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id']
train_dataset = BatchGenerator(labels_path=args.labels, input_format=labels_format, include_classes=args.classes)

training = train_dataset.generate(args.batch_size, ssd_box_encoder=ssd_box_encoder)

val_dataset = BatchGenerator(args.test if args.test else args.labels, labels_format, include_classes=args.classes)

validation = val_dataset.generate(args.batch_size, ssd_box_encoder=ssd_box_encoder)


# 7: Define a simple learning rate schedule
def lr_schedule(epoch):
    if epoch <= 20:
        return 0.01
    elif epoch <= 200:
        return 0.001
    else:
        return 0.0001


print(val_dataset.count, ceil(val_dataset.count / args.batch_size))
history = model.fit_generator(generator=training,
                              steps_per_epoch=ceil(train_dataset.count / args.batch_size),
                              epochs=args.epochs,
                              callbacks=[ModelCheckpoint('./{}_weights.h5'.format(args.model),
                                                         monitor='val_loss',
                                                         verbose=1,
                                                         save_best_only=False,
                                                         save_weights_only=True),
                                         LearningRateScheduler(lr_schedule)],
                              validation_data=validation,
                              validation_steps=ceil(val_dataset.count / args.batch_size))

model_name = 'ssd300_0'
model.save('./{}.h5'.format(model_name))
model.save_weights('./{}_weights.h5'.format(model_name))

print()
print("Model saved as {}.h5".format(model_name))
print("Weights also saved separately as {}_weights.h5".format(model_name))


if args.val_dir:
    results = []

    for r, d, filenames in os.walk(args.val_dir):
        for filename in filenames:
            if filename.endswith('png'):
                with rasterio.open(os.path.join(r, filename)) as f:
                    x = f.read().transpose([1, 2, 0])[np.newaxis, :]
                    p = decode_y2(model.predict(x),
                                  confidence_thresh=0.15,
                                  iou_threshold=0.35,
                                  top_k=200,
                                  input_coords='minmax',
                                  normalize_coords=normalize_coords,
                                  img_height=img_height,
                                  img_width=img_width)
                    for row in p[0]:
                        results.append([filename] + row.tolist())
    pandas.DataFrame(results, columns=['file_name', 'class_id', 'conf', 'xmin', 'xmax', 'ymin', 'ymax']).to_csv(
        'ssd_results.csv')
