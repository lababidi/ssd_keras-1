import os
from argparse import ArgumentParser
from math import ceil

import geopandas
import h5py
import numpy as np
import pandas
import rasterio
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam

# from singleshot import SSD, SSDLoss, SSDBoxEncoder, BatchGenerator, decode_y
import singleshot

def append_to_aspect_ratio_list(max_aspect_ratio=None):
    ratios = list(1 / np.linspace(1.0, max_aspect_ratio, 6))
    ratios.reverse()
    ratios += list(np.linspace(1.0, max_aspect_ratio, 6))[1:]

    return [ratios] * 6


def console():
    parser = get_args()
    parser.add_argument('csv', default='/osn/share/rail.csv')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    print(args.min_scale, args.max_scale)
    img_height = 300  # Height of the input images
    img_width = 300  # Width of the input images
    img_channels = 3  # Number of color channels of the input images
    n_classes = len(args.classes) + 1 if args.classes else 2
    if '0' in args.classes:
        n_classes -= 1
    scales = [args.scale] * 7 if args.scale else None
    aspect_ratios = [[1.0]] * 6
    if args.max_aspect:
        aspect_ratios = append_to_aspect_ratio_list(args.max_aspect)
    two_boxes_for_ar1 = True
    limit_boxes = False
    variances = [0.1, 0.1, 0.2, 0.2]
    coords = 'minmax'
    normalize_coords = False

    K.clear_session()
    model, predictor_sizes = singleshot.SSD(image_size=(img_height, img_width, img_channels),
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
                                 normalize_coords=normalize_coords,
                                 max_pixel=args.max_pixel)
    if args.model:
        model.load_weights(args.model, by_name=True)

    model.compile(optimizer=(Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)),
                  loss=singleshot.SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=0.1).compute_loss)

    ssd_box_encoder = singleshot.SSDBoxEncoder(img_height=img_height,
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

    dataset_generator = singleshot.BatchGenerator(include_classes=args.classes)

    if not os.path.exists(args.name):
        os.mkdir(args.name)

    dataset_generator.parse_csv(labels_path=args.csv,
                                input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                                split_ratio=args.split_ratio,
                                checkpoints_path=args.name)

    train_generator = dataset_generator.generate(
        batch_size=args.batch_size,
        train=True,
        ssd_box_encoder=ssd_box_encoder,
        rgb_to_gray=args.rgb_to_gray,
        gray_to_rgb=args.gray_to_rgb,
        multispectral_to_rgb=args.multispectral_to_rgb,
        hist=args.hist,
    )

    val_generator = dataset_generator.generate(
        batch_size=args.batch_size,
        train=True,
        ssd_box_encoder=ssd_box_encoder,
        rgb_to_gray=args.rgb_to_gray,
        gray_to_rgb=args.gray_to_rgb,
        multispectral_to_rgb=args.multispectral_to_rgb,
        val=True,
        hist=args.hist,
    )

    def lr_schedule(epoch):
        if epoch <= 500:
            # return 0.01
            return 0.0001
        else:
            return 0.00001

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

    val_dir = '/osn/SpaceNet-MOD/testing/rgb-ps-dra/300/'

    results = []
    for r, d, filenames in os.walk(val_dir):
        for filename in filenames:
            if filename.endswith('png'):
                with rasterio.open(os.path.join(r, filename)) as f:
                    x = f.read().transpose([1, 2, 0])[np.newaxis, :]
                    p = model.predict(x)
                    try:
                        y = singleshot.decode_y(p,
                                     confidence_thresh=0.15,
                                     iou_threshold=0.35,
                                     top_k=200,
                                     input_coords='minmax',
                                     normalize_coords=normalize_coords,
                                     img_height=img_height,
                                     img_width=img_width)
                        for row in y[0]:
                            results.append([filename] + row.tolist())
                    except ValueError:
                        pass
    df = pandas.DataFrame(results, columns=['file_name', 'class_id', 'conf', 'xmin', 'xmax', 'ymin', 'ymax'])
    df['class_id'] = df['class_id'].apply(lambda xx: dataset_generator.class_map_inv[xx])
    df.to_csv('./' + args.name + '/' + args.outcsv)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--name', default='ssd')
    parser.add_argument('--classes', type=lambda ss: [int(s) for s in ss.split(',')])
    parser.add_argument('--scale', type=float)
    parser.add_argument('--min_scale', type=float)
    parser.add_argument('--max_scale', type=float)
    parser.add_argument('--max_aspect', type=float)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--rgb_to_gray', dest='rgb_to_gray', action="store_true")
    parser.add_argument('--gray_to_rgb', dest='gray_to_rgb', action="store_true")
    parser.add_argument('--multispectral_to_rgb', type=bool, default=False)
    parser.add_argument('--hist', dest='hist', action="store_true")
    parser.add_argument('--max_pixel', type=float, default=255.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--outcsv', default='ssd_results.csv')
    parser.add_argument('--split_ratio', type=float, default=1.0)
    parser.add_argument('--gpus', default='0,1,2,3')
    return parser


def validate():
    parser = get_args()
    parser.add_argument('--image_size', default=300)
    parser.add_argument('input', help="Location of input images")
    parser.add_argument('output', help="Location of pb output")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    print(args.min_scale, args.max_scale)
    img_height = 300  # Height of the input images
    img_width = 300  # Width of the input images
    img_channels = 3  # Number of color channels of the input images
    n_classes = len(args.classes) + 1 if args.classes else 2
    if '0' in args.classes:
        n_classes -= 1
    scales = [args.scale] * 7 if args.scale else None
    aspect_ratios = [[1.0]] * 6
    if args.max_aspect:
        aspect_ratios = append_to_aspect_ratio_list(args.max_aspect)
    two_boxes_for_ar1 = True
    limit_boxes = False
    variances = [0.1, 0.1, 0.2, 0.2]
    coords = 'minmax'
    normalize_coords = False

    K.clear_session()
    model, predictor_sizes = singleshot.SSD(image_size=(img_height, img_width, img_channels),
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
                                 normalize_coords=normalize_coords,
                                 max_pixel=args.max_pixel)
    if args.model:
        model.load_weights(args.model, by_name=True)

    # model = keras.models.load_model(args.model,
    #                                 custom_objects={"tf": tf,
    #                                                 'L2Normalization': L2Normalization,
    #                                                 "AnchorBoxes": AnchorBoxes})

    import cv2
    from shapely.geometry import Polygon
    from shapely.affinity import affine_transform
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    class_map_inv = {k + 1: v for k, v in enumerate(args.classes)} if args.classes else None

    results = []
    for r, d, filenames in os.walk(args.input):
        for filename in filenames:
            print(filename)
            with rasterio.open(os.path.join(r, filename)) as f:
                x = f.read().transpose([1, 2, 0])
                print(f.transform)
                if args.gray_to_rgb:
                    if args.hist:
                        x = clahe.apply(x)
                    x = cv2.cvtColor(x, cv2.COLOR_BayerGR2RGB)
                x = x[np.newaxis, :]
                p = model.predict(x)
                try:
                    y = singleshot.decode_y(p,
                                 confidence_thresh=0.15,
                                 iou_threshold=0.35,
                                 top_k=2000,
                                 input_coords='minmax',
                                 normalize_coords=normalize_coords,
                                 img_height=img_height,
                                 img_width=img_width)
                    print("Found {} results".format(len(y[0])))
                    for box in y[0]:
                        poly = Polygon([[box[2], box[4]], [box[3], box[4]], [box[3], box[5]],
                                              [box[2], box[5]], [box[2], box[4]]])
                        results.append((filename, box[0], affine_transform(poly, f.transform[:6]), box[1]))
                        # results.append([filename] + row.tolist())
                except ValueError as e:
                    print(e)
                    continue
    df = geopandas.GeoDataFrame(results, columns=['file_name', 'class_id', 'geometry', 'conf'])
    # df = pandas.DataFrame(results, columns=['file_name', 'class_id', 'conf', 'xmin', 'xmax', 'ymin', 'ymax'])
    df['class_id'] = df['class_id'].apply(lambda xx: class_map_inv[xx])
    print(df.head())
    print(df.dtypes)
    with open(args.output) as f:
        f.write(df.to_json())
    # df.to_csv(args.output)


def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape))
    finally:
        f.close()


def convert_model():
    parser = get_args()
    parser.add_argument('output', help="Location of pb output")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    n_classes = len(args.classes) + 1 if args.classes else 2
    scales = [args.scale] * 7 if args.scale else None
    aspect_ratios = [[1.0]] * 6
    two_boxes_for_ar1 = True
    limit_boxes = False
    variances = [0.1, 0.1, 0.2, 0.2]
    coords = 'minmax'
    normalize_coords = False

    model, predictor_sizes = singleshot.SSD(image_size=(args.image_size, args.image_size, 3),
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
                                 normalize_coords=normalize_coords,
                                 max_pixel=args.max_pixel)

    model.load_weights(args.model)

    from tensorflow.python.framework import graph_io
    from tensorflow.python.framework import graph_util

    # Export a serialized graph
    sess = K.get_session()
    tf.identity(model.outputs[0], name='output_node0')
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['output_node0'])
    graph_io.write_graph(constant_graph, args.output, 'ssd-model.pb', as_text=False)

    import subprocess
    subprocess.run([
        'gbdxm', 'pack',
        '--gbdxm-file', 'ssd-model.gbdxm',
        '--name', 'ssd-model',
        '--version', '0',
        '--description', "ssd-model",
        '--model-size', '300', '300',  # Match SSD.image_size[0,1] \
        '--color-mode', 'rgb',  # Match SSD.image_size[2] \
        '--label-names', 'a b c d e f g h i',  # One for each SSD.n_classes \
        '--tensorflow-model', os.path.join(args.output, 'ssd-model.pb'),  # Match graph_io.write_graph path \
        '--tensorflow-input-layer', 'input_1',
        '--tensorflow-output-layers', 'output_node0',  # Match tf.identity name \
        '--tensorflow-output-space', 'pixel',  # Match SSD.normalize_coords \
        '--tensorflow-output-type', 'minmax',  # Match SSD.coords \
        '--type', 'tensorflow',
        '--category', 'ssd'
    ])