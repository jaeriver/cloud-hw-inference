#!/usr/bin/env python
# coding: utf-8

# # Evaluate YOLO v3 on Inferentia
# ## Note: this tutorial runs on tensorflow-neuron 1.x only

# ## Introduction
# This tutorial walks through compiling and evaluating YOLO v3 model on Inferentia using the AWS Neuron SDK.
# 
# 
# In this tutorial we provide two main sections:
# 
# 1. Download Dataset and Generate Pretrained SavedModel
# 
# 2. Compile the YOLO v3 model.
# 
# 3. Deploy the same compiled model.
# 
# Before running the following verify this Jupyter notebook is running “conda_aws_neuron_tensorflow_p36” kernel. You can select the Kernel from the “Kernel -> Change Kernel” option on the top of this Jupyter notebook page.
# 
# Instructions of how to setup Neuron Tensorflow environment and run the tutorial as a Jupyter notebook are available in the Tutorial main page [Tensorflow-YOLO_v3 Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/tensorflow-neuron/tutorials/yolo_v3_demo/yolo_v3_demo.html)

# ## Prerequisites
# 

# This demo requires the following pip packages:
# 
# `pillow matplotlib pycocotools`
# 

# ## Deploy the model on Inferentia
# ## Part 3:Evaluate Model Quality after Compilation
# 
# ### Define evaluation functions
# We first define some handy helper functions for running evaluation on the COCO 2017 dataset.

# In[1]:


import os
import json
import time
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import pandas as pd

def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000)):
    """
    Args:
        jsonfile: Evaluation json file, eg: bbox.json, mask.json.
        style: COCOeval style, can be `bbox` , `segm` and `proposal`.
        coco_gt: Whether to load COCOAPI through anno_file,
                 eg: coco_gt = COCO(anno_file)
        anno_file: COCO annotations file.
        max_dets: COCO evaluation maxDets.
    """
    assert coco_gt is not None or anno_file is not None

    if coco_gt is None:
        coco_gt = COCO(anno_file)
    print("Start evaluate...")
    coco_dt = coco_gt.loadRes(jsonfile)
    if style == 'proposal':
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.useCats = 0
        coco_eval.params.maxDets = list(max_dets)
    else:
        coco_eval = COCOeval(coco_gt, coco_dt, style)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def bbox_eval(anno_file, bbox_list):
    coco_gt = COCO(anno_file)

    outfile = 'bbox_detections.json'
    print('Generating json file...')
    with open(outfile, 'w') as f:
        json.dump(bbox_list, f)

    map_stats = cocoapi_eval(outfile, 'bbox', coco_gt=coco_gt)
    return map_stats


def get_image_as_bytes(images, eval_pre_path, user_batch_size):
    batch_im_id_list = []
    batch_im_name_list = []
    batch_img_bytes_list = []
    n = len(images)
    batch_im_id = []
    batch_im_name = []
    batch_img_bytes = []
    for i, im in enumerate(images):
        im_id = im['id']
        file_name = im['file_name']
        if i % user_batch_size == 0 and i != 0:
            batch_im_id_list.append(batch_im_id)
            batch_im_name_list.append(batch_im_name)
            batch_img_bytes_list.append(batch_img_bytes)
            batch_im_id = []
            batch_im_name = []
            batch_img_bytes = []
        batch_im_id.append(im_id)
        batch_im_name.append(file_name)

        with open(os.path.join(eval_pre_path, file_name), 'rb') as f:
            batch_img_bytes.append(f.read())
    return batch_im_id_list, batch_im_name_list, batch_img_bytes_list


def analyze_bbox(results, batch_im_id, _clsid2catid):
    bbox_list = []
    k = 0
    for boxes, scores, classes in zip(results['boxes'], results['scores'], results['classes']):
        if boxes is not None:
            im_id = batch_im_id[k]
            n = len(boxes)
            for p in range(n):
                clsid = classes[p]
                score = scores[p]
                xmin, ymin, xmax, ymax = boxes[p]
                catid = (_clsid2catid[int(clsid)])
                w = xmax - xmin + 1
                h = ymax - ymin + 1

                bbox = [xmin, ymin, w, h]
                # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
                bbox = [round(float(x) * 10) / 10 for x in bbox]
                bbox_res = {
                    'image_id': im_id,
                    'category_id': catid,
                    'bbox': bbox,
                    'score': float(score),
                }
                bbox_list.append(bbox_res)
        k += 1
    return bbox_list


# Here is the actual evaluation loop. To fully utilize all four cores on one Inferentia, the optimal setup is to run multi-threaded inference using a `ThreadPoolExecutor`. The following cell is a multi-threaded adaptation of the evaluation routine at https://github.com/miemie2013/Keras-YOLOv4/blob/910c4c6f7265f5828fceed0f784496a0b46516bf/tools/cocotools.py#L97.

# In[2]:


from concurrent import futures

def evaluate(yolo_predictor, images, eval_pre_path, anno_file, eval_batch_size, _clsid2catid, user_batch_size, num_cores):
    batch_im_id_list, batch_im_name_list, batch_img_bytes_list = get_image_as_bytes(images, eval_pre_path, user_batch_size)
    walltime_start = time.time()
    # warm up
#     yolo_predictor({'image': np.array(batch_img_bytes_list[0], dtype=object)})

#     with futures.ThreadPoolExecutor(4) as exe:
#         fut_im_list = []
#         fut_list = []
#         start_time = time.time()
#         for batch_im_id, batch_im_name, batch_img_bytes in zip(batch_im_id_list, batch_im_name_list, batch_img_bytes_list):
#             if len(batch_img_bytes) != eval_batch_size:
#                 continue
#             fut = exe.submit(yolo_predictor, {'image': np.array(batch_img_bytes, dtype=object)})
#             fut_im_list.append((batch_im_id, batch_im_name))
#             fut_list.append(fut)
#         bbox_list = []
#         count = 0
#         for (batch_im_id, batch_im_name), fut in zip(fut_im_list, fut_list):
#             results = fut.result()
#             bbox_list.extend(analyze_bbox(results, batch_im_id, _clsid2catid))
#             for _ in batch_im_id:
#                 count += 1
#                 if count % 100 == 0:
#                     print('Test iter {}'.format(count))
#         print('==================== Performance Measurement ====================')
#         print('Finished inference on {} images in {} seconds'.format(len(images), time.time() - start_time))
#         print('=================================================================')
#     # start evaluation
#     box_ap_stats = bbox_eval(anno_file, bbox_list)
    iter_times = []
    counter = 0
    first_iter_time = 0
    fut_im_list = []
    fut_list = []
    for batch_im_id, batch_im_name, batch_img_bytes in zip(batch_im_id_list, batch_im_name_list, batch_img_bytes_list):
        if len(batch_img_bytes) != user_batch_size:
            continue
        iter_start = time.time()
        fut = yolo_predictor({'image': np.array(batch_img_bytes, dtype=object)})
        fut_im_list.append((batch_im_id, batch_im_name))
        fut_list.append(fut)
        if counter == 0:
            first_iter_time = time.time() - iter_start
        else:
            iter_times.append(time.time() - iter_start)
        counter +=1
    bbox_list = []
    counter = 0
    for (batch_im_id, batch_im_name), fut in zip(fut_im_list, fut_list):
        results = fut
        bbox_list.extend(analyze_bbox(results, batch_im_id, _clsid2catid))
        for _ in batch_im_id:
            counter += 1
            if counter % 100 == 0:
                print('Test iter {}'.format(counter))
    
    print('==================== Performance Measurement ====================')
    print('Finished inference on {} images in {} seconds'.format(len(images), time.time() - walltime_start))
    print('=================================================================')
    
    results = pd.DataFrame(columns = [f'inf1_compiled_batch_size_{eval_batch_size}_compiled_cores_{num_cores}'])
    results.loc['compiled_batch_size'] = [eval_batch_size]
    results.loc['user_batch_size'] = [user_batch_size]
    results.loc['first_prediction_time'] = [first_iter_time]
    results.loc['average_prediction_time'] = [np.mean(iter_times)]
    results.loc['wall_time'] = [time.time() - walltime_start]
    box_ap_stats = bbox_eval(anno_file, bbox_list)
    return box_ap_stats, results, iter_times


# ### Evaluate mean average precision (mAP) score
# Here is the code to calculate mAP scores of the YOLO v3 model. The expected mAP score is around 0.328 if we use the pretrained weights.

# In[32]:


import glob
from PIL import Image

def filenames_to_input(file_list):
    imgs = []
    for file in file_list:
        img = Image.open(file)
        img.convert('RGB')
        img = img.resize((640, 640), Image.ANTIALIAS)
        img = np.array(img, dtype='float32')
        # if image is grayscale, convert to 3 channels
        if len(img.shape) != 3:
            img = np.repeat(img[..., np.newaxis], 3, -1)
        # batchsize, 224, 224, 3
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        imgs.append(img)

    batch_imgs = np.vstack(imgs)
    return batch_imgs

from tensorflow.keras.models import load_model

val_coco_root = './val2017'
val_annotate = './annotations/instances_val2017.json'
clsid2catid = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16,
               15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31,
               27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43,
               39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56,
               51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72,
               63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85,
               75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

model_type = 'yolo_v5_coco'

batch_list = [1]
num_of_cores = [1]
user_batchs = [1]
inf1_model_dir = f'{model_type}_inf1_saved_models'
for user_batch in user_batchs:
    iter_ds = pd.DataFrame()
    results = pd.DataFrame()
    for eval_batch_size in batch_list:
        for num_cores in num_of_cores:
            opt ={'batch_size': eval_batch_size, 'num_cores': num_of_cores}
#             compiled_model_dir = f'{model_type}_batch_{eval_batch_size}_inf1_cores_{num_cores}'
#             inf1_compiled_model_dir = os.path.join(inf1_model_dir, compiled_model_dir)
            inf1_compiled_model_dir = inf1_model_dir
            print(f'inf1_compiled_model_dir: {inf1_compiled_model_dir}')
            col_name = lambda opt: f'inf1_{eval_batch_size}_multicores_{num_cores}'

            with open(val_annotate, 'r', encoding='utf-8') as f2:
                for line in f2:
                    line = line.strip()
                    dataset = json.loads(line)
                    images = dataset['images']
            start_time = time.time()
            yolo_pred = load_model(inf1_compiled_model_dir)
            load_time = time.time() - start_time
            iter_times = []
            
            image_list = glob.glob(val_coco_root + '/*')
            for image in image_list:
                image = filenames_to_input([image])
                start_time = time.time()
                res = yolo_pred(image)
                iter_times.append(time.time() - start_time)
                break
            
            iter_times = np.array(iter_times)
            
            results = pd.DataFrame(columns = [f'inf1_tf2_{model_type}_{1}'])
#             results.loc['batch_size']              = [batch_size]
#             results.loc['first_prediction_time']   = [first_iter_time]
            results.loc['average_prediction_time'] = [np.mean(iter_times)]
            results.loc['load_time']               = [load_time]
#             box_ap, res, iter_times = evaluate(yolo_pred,
#                                                images,
#                                                val_coco_root,
#                                                val_annotate,
#                                                eval_batch_size,
#                                                clsid2catid,
#                                                eval_batch_size * user_batch, 
#                                                num_cores)

#         iter_ds = pd.concat([iter_ds, pd.DataFrame(iter_times, columns=[col_name(opt)])], axis=1)
#         results = pd.concat([results, res], axis=1)
#     display(results)
print(results)


# In[ ]:




