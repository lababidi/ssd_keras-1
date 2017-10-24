import os
from argparse import ArgumentParser
from math import ceil

import numpy as np
import pandas
import rasterio
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam

from singleshot import SSD, SSDLoss
from singleshot.util import BatchGenerator, decode_y, SSDBoxEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def console():
    parser = ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--name', default='ssd')
    parser.add_argument('--scale', type=float)
    parser.add_argument('--classes', type=lambda ss: [int(s) for s in ss.split(',')])
    parser.add_argument('--min_scale', type=float)
    parser.add_argument('--max_scale', type=float)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--outcsv', default='ssd_results.csv')
    parser.add_argument('--split_ratio', type=float, default=1.0)
    parser.add_argument('csv', default='/osn/share/rail.csv')
    args = parser.parse_args()

    def appent_to_aspect_ratio_list(aspect_ratios = None,
                              max_aspect_ratio = None):

        ratios = list(1 / np.linspace(1.0, max_aspect_ratio, 6))
        ratios.reverse()
        ratios += list(np.linspace(1.0, max_aspect_ratio, 6))[1:]

        aspect_ratios.append(ratios)

    print(args.min_scale, args.max_scale)
    img_height = 300  # Height of the input images
    img_width = 300  # Width of the input images
    img_channels = 3  # Number of color channels of the input images
    n_classes = len(args.classes)+1 if args.classes else 2
    # Number of classes including the background class, e.g. 21 for the Pascal VOC datasets
    scales = [0.01, 0.04, 0.07, 0.1, 0.13, 0.17, 0.5 ] #[args.scale] * 7 if args.scale else None
    aspect_ratios = [] #[[1.0]] * 6
    appent_to_aspect_ratio_list(aspect_ratios, 15.0)
    appent_to_aspect_ratio_list(aspect_ratios, 12.0)
    appent_to_aspect_ratio_list(aspect_ratios, 10.0)
    appent_to_aspect_ratio_list(aspect_ratios, 6.0)
    appent_to_aspect_ratio_list(aspect_ratios, 4.5)
    appent_to_aspect_ratio_list(aspect_ratios, 4.0)
    two_boxes_for_ar1 = True
    limit_boxes = False
    variances = [0.1, 0.1, 0.2, 0.2]
    coords = 'minmax'
    normalize_coords = False

    K.clear_session()
    model, predictor_sizes = SSD(image_size=(img_height, img_width, img_channels),
                                     n_classes=n_classes,
                                     min_scale=args.min_scale,
                                     max_scale=args.max_scale,
                                     scales=scales,
                                     aspect_ratios_global=None,
                                     aspect_ratios_per_layer=aspect_ratios,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     coords=coords,
                                     normalize_coords=normalize_coords)
    if args.model:
        model.load_weights(args.model, by_name=True)


    model.compile(optimizer=(Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)),
                  loss=SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=0.1).compute_loss)


    ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    min_scale=args.min_scale,
                                    max_scale=args.max_scale,
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


    dataset_generator = BatchGenerator(include_classes=args.classes)

    dataset_generator.parse_csv(labels_path=args.csv,
                            input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], split_ratio=args.split_ratio)

    train_generator = dataset_generator.generate(batch_size=args.batch_size,
                                             train=True,
                                             ssd_box_encoder=ssd_box_encoder,
                                             limit_boxes=True,  # While the anchor boxes are not being clipped,
                                             include_thresh=0.4,
                                             diagnostics=False)

    val_generator = dataset_generator.generate(batch_size=args.batch_size,
                                         train=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         equalize=False,
                                         brightness=False,
                                         flip=False,
                                         translate=False,
                                         scale=False,
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True,
                                         include_thresh=0.4,
                                         diagnostics=False,
                                         val=True)

    def lr_schedule(epoch):
        if epoch <= 500:
            return 0.001
        else:
            return 0.0001

    if not os.path.exists(args.name):
        os.mkdir(args.name)

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=ceil(dataset_generator.count / args.batch_size),
                                  epochs=args.epochs,
                                  callbacks=[ModelCheckpoint('./' + args.name + '/epoch{epoch:04d}_loss{loss:.4f}.h5',
                                                             monitor='val_loss',
                                                             verbose=1,
                                                             save_best_only=False,
                                                             save_weights_only=False,
                                                             mode='auto',
                                                             period=1),
                                             LearningRateScheduler(lr_schedule),
                                             ],
                                  validation_data=val_generator,
                                  validation_steps=ceil(dataset_generator.count / args.batch_size))

    model.save('./' + args.name + '/{}.h5'.format(args.name))
    model.save_weights('./' + args.name + '/{}_weights.h5'.format(args.name))

    print("Model and weights saved as {}[_weights].h5".format(args.name))

    predict_generator = dataset_generator.generate(batch_size=1,
                                             train=False,
                                             equalize=False,
                                             brightness=False,
                                             flip=False,
                                             translate=False,
                                             scale=False,
                                             crop=False,
                                             resize=False,
                                             gray=False,
                                             limit_boxes=True,
                                             include_thresh=0.4,
                                             diagnostics=False,
                                             val=True)


    val_dir = '/osn/SpaceNet-MOD/testing/rgb-ps-dra/300/'

    results = []
    for r, d, filenames in os.walk(val_dir):
        for filename in filenames:
            if filename.endswith('png'):
                with rasterio.open(os.path.join(r, filename)) as f:
                    x = f.read().transpose([1, 2, 0])[np.newaxis, :]
                    p = model.predict(x)
                    try:
                        y = decode_y(p,
                                     confidence_thresh=0.15,
                                     iou_threshold=0.35,
                                     top_k=200,
                                     input_coords='minmax',
                                     normalize_coords=normalize_coords,
                                     img_height=img_height,
                                     img_width=img_width)
                        for row in y[0]:
                            results.append([filename] + row.tolist())
                    except ValueError as e:
                        pass
    df = pandas.DataFrame(results, columns=['file_name', 'class_id', 'conf', 'xmin', 'xmax', 'ymin', 'ymax'])
    df['class_id'] = df['class_id'].apply(lambda xx: dataset_generator.class_map_inv[xx])
    df.to_csv('./' + args.name + '/' + args.outcsv)
