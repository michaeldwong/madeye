import shutil
import random
import evaluation_tools
import statistics
import copy
import json
import os
import torch
import pandas as pd
import train
import numpy as np
from torch import nn
from madeye_utils import parse_orientation_string, extract_pan, extract_tilt, extract_zoom, find_tilt_dist, find_pan_dist


from torch.utils.data import DataLoader
from torchvision import transforms
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string
from tqdm.autonotebook import tqdm
from zoom_explore_helper import add_zoom_factors, reset_zoom_factors
from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string




MODELS = ['yolov4', 'ssd-voc', 'tiny-yolov4', 'faster-rcnn']
SKIP = 6 # Only consider frames where frame % SKIP == 0
PERSON_CONFIDENCE_THRESH = 50.0
CAR_CONFIDENCE_THRESH = 70.0


nms_threshold = 0.5
model_params = train.Params(f'projects/madeye.yml')
obj_list = model_params.obj_list
compound_coef = 0
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


use_float16 = False


def generate_plus_formation(current_orientation):
    items = current_orientation.split('-')
    pan = int(items[0])
    zoom = int(items[-1])
    if pan == 0:
        left_horz = 330
    else:
        left_horz = int(items[0]) - 30
    if pan == 330:
        right_horz = 0
    else:
        right_horz = int(items[0]) + 30

    if len(items) == 4:
        tilt = int(items[2]) * -1
    else:
        tilt = int(items[1])
    top_tilt = tilt + 15
    bottom_tilt = tilt - 15

    if tilt == 30:
        return [ f'{left_horz}-{tilt}-{zoom}',
                 f'{right_horz}-{tilt}-{zoom}',
                 current_orientation,
                 f'{pan}-{bottom_tilt}-{zoom}' ]
    elif tilt == -30:
        return [ f'{left_horz}-{tilt}-{zoom}',
                 f'{right_horz}-{tilt}-{zoom}',
                 current_orientation,
                 f'{pan}-{top_tilt}-{zoom}']
    return [ f'{left_horz}-{tilt}-{zoom}',
             f'{right_horz}-{tilt}-{zoom}',
             current_orientation,
             f'{pan}-{top_tilt}-{zoom}',
             f'{pan}-{bottom_tilt}-{zoom}' ]

def rank_orientations(orientation_to_count):
    sorted_dict = {k: v for k, v in sorted(orientation_to_count.items(), key=lambda item: item[1] * -1)}
    orientation_to_rank = {}
    last_count = 0
    rank = 0
    for o in sorted_dict:
        count = sorted_dict[o]
        if count != last_count:
            last_count = count
            rank += 1
        if rank == 0:
            rank += 1
        orientation_to_rank[o] = rank
    return orientation_to_rank

class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def create_annotations(f, image_file, orientation_df, orientation, object_type, json_dict, image_id, annotation_id):
    if object_type != 'car' and object_type != 'person' and object_type != 'both':
        raise Exception('Incorrect object type')
    count = 0
    for idx, row in orientation_df.iterrows():
        if object_type == 'car' or object_type == 'both':
            if row['class'] == 'car' and row['confidence'] >= CAR_CONFIDENCE_THRESH:
                xmin = row['left']
                xmax = row['right']
                ymin = row['top']
                ymax = row['bottom']
                json_dict['annotations'].append({"id": annotation_id,"image_id": image_id, "category_id": 1, "iscrowd": 0, "image_id": image_id, "bbox": [xmin, ymin, xmax - xmin, ymax - ymin ], "area": (ymax - ymin) * (xmax - xmin), "segmentation": [[xmin, ymin, xmax , ymin, xmax, ymax, xmin, ymax ]] })
        if object_type == 'person' or object_type == 'both':
            if row['class'] == 'person' and row['confidence'] >= PERSON_CONFIDENCE_THRESH:
                xmin = row['left']
                xmax = row['right']
                ymin = row['top']
                ymax = row['bottom']
                json_dict['annotations'].append({"id": annotation_id, "image_id": image_id, "category_id": 2, "iscrowd": 0, "image_id": image_id, "bbox": [xmin, ymin, xmax - xmin, ymax - ymin ], "area": (ymax - ymin) * (xmax - xmin), "segmentation": [[xmin, ymin, xmax , ymin, xmax, ymax, xmin, ymax ]] })
        annotation_id += 1
    return annotation_id

# set_type is 'train' or 'val'
def generate_dataset(inference_dir, rectlinear_dir, model_name, current_frame, orientation_to_frames, set_type, data_path,  project_name):
    json_dict = {
        "info": {
            "description": "","url": "","version": "1.0","year": 2017,"contributor": "","date_created": "2017/09/01"
        },
        "licenses": [
                {"id": 1, "name": "None", "url": "None"}
        ],
        "images": [

        ],
        "annotations": [

        ],
        "categories": [
            {"id": 1, "name": "car", "supercategory": "None"},
            {"id": 2, "name": "person", "supercategory": "None"},
        ],
    }

    image_id = 0
    annotation_id = 0
    image_outdir = f'{data_path}/{project_name}/{model_name}/{set_type}'
#    if os.path.exists(image_outdir):
#        shutil.rmtree(image_outdir)
    annotations_outdir = f'{data_path}/{project_name}/{model_name}/annotations/'
#    if os.path.exists(os.path.join(annotations_outdir, f'instances_{set_type}.json')):
#        shutil.rmtree(os.path.join(annotations_outdir, f'instances_{set_type}.json'))
    os.makedirs(annotations_outdir, exist_ok=True)
    os.makedirs(image_outdir, exist_ok=True)

    print('generating dataset')
    print(orientation_to_frames)
    for o in orientation_to_frames:
        frames = orientation_to_frames[o]
        result_orientation_dir = os.path.join(inference_dir, model_name, o)
        for f in frames:
            inference_file = os.path.join(result_orientation_dir, f'frame{f}.csv')
            if os.path.getsize(inference_file) > 0:
                orientation_df = pd.read_csv(inference_file)
                orientation_df.columns = ['left', 'top', 'right', 'bottom', 'class', 'confidence']
                orig_image_file = os.path.join(rectlinear_dir, o, f'frame{current_frame}.jpg')
                image_file = f'{o}-frame{current_frame}.jpg'
                dest = f'{image_outdir}/{image_file}'
                shutil.copy(orig_image_file, dest)
                json_dict['images'].append({"id": image_id, "file_name": image_file, "width": 1280, "height": 720, "date_captured": "", "license": 1, "coco_url": "", "flickr_url": ""})
                annotation_id = create_annotations(f, image_file, orientation_df, o, 'both', json_dict, image_id, annotation_id)
            image_id += 1

    with open(os.path.join(annotations_outdir , f'instances_{set_type}.json'), 'w') as f_out:
        json.dump(json_dict, f_out)


def continual_train( model_name, weights_path, data_path, saved_path, project_name, gpu, num_epochs=10):

    use_cuda = gpu >= 0
    # Hardcoded options
    batch_size = 4
    compound_coef = 0
    # Input to dataset
    head_only = True
    val_interval = 1
    es_min_delta = 0.0
    num_workers = 12
    lr = 1e-4
    optim = 'adamw'
    es_patience = 0
    save_interval = 100
  
    best_weights = weights_path
 
    os.makedirs(os.path.join(data_path, project_name, model_name), exist_ok=True)
    os.makedirs(os.path.join(saved_path, project_name, model_name), exist_ok=True)


    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    if gpu >= 0:
        torch.cuda.set_device(f'cuda:{gpu}')



    training_params = {'batch_size': batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': num_workers}

    val_params = {'batch_size': batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    print('Reading from ', os.path.join(data_path, project_name))
    print('train set ', model_params.train_set)
    training_set = CocoDataset(root_dir=os.path.join(data_path, project_name, model_name), set=model_params.train_set,
                               transform=transforms.Compose([Normalizer(mean=model_params.mean, std=model_params.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[compound_coef])]))
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=os.path.join(data_path, project_name, model_name), set=model_params.val_set,
                          transform=transforms.Compose([Normalizer(mean=model_params.mean, std=model_params.std),
                                                        Resizer(input_sizes[compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(model_params.obj_list), compound_coef=compound_coef,
                                 ratios=eval(model_params.anchors_ratios), scales=eval(model_params.anchors_scales))

    try:
        
        if gpu >= 0:
            ret = model.load_state_dict(torch.load(weights_path  ),strict=False)
            model.to(f'cuda:{gpu}')
        else:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    except RuntimeError as e:
        print(f'[Warning] Ignoring {e}')
        print(
            '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

    print(f'[Info] loaded weights: {os.path.basename(weights_path)}')
    # freeze backbone if train head_only
    if head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if model_params.num_gpus > 1 and batch_size // model_params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model)

    if model_params.num_gpus > 0:
        model = model.cuda()
        if model_params.num_gpus > 1:
            model = CustomDataParallel(model, model_params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    model.train()

    step = 0
    num_iter_per_epoch = len(training_generator)
    min_loss = 0.0
    try:
        for epoch in range(num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']
                    if model_params.num_gpus == 1 and gpu >= 0:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=model_params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()
                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
#                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % save_interval == 0 and step > 0:
                        save_checkpoint_continual_learning(model, f'efficientdet-d{compound_coef}_{epoch}.pth', os.path.join(saved_path, project_name, model_name))
                    if loss < min_loss:
                        save_checkpoint_continual_learning(model, f'efficientdet-d{compound_coef}_min.pth', os.path.join(saved_path, project_name, model_name))

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            # VAlidation testing
#            if epoch % val_interval == 0:
#                model.eval()
#                loss_regression_ls = []
#                loss_classification_ls = []
#                for iter, data in enumerate(val_generator):
#                    with torch.no_grad():
#                        imgs = data['img']
#                        annot = data['annot']
#
#                        if model_params.num_gpus == 1:
#                            imgs = imgs.cuda()
#                            annot = annot.cuda()
#
#                        cls_loss, reg_loss = model(imgs, annot, obj_list=model_params.obj_list)
#                        cls_loss = cls_loss.mean()
#                        reg_loss = reg_loss.mean()
#
#                        loss = cls_loss + reg_loss
#                        if loss == 0 or not torch.isfinite(loss):
#                            continue
#
#                        loss_classification_ls.append(cls_loss.item())
#                        loss_regression_ls.append(reg_loss.item())
#
#                cls_loss = np.mean(loss_classification_ls)
#                reg_loss = np.mean(loss_regression_ls)
#                loss = cls_loss + reg_loss
#
#                print(
#                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
#                        epoch, num_epochs, cls_loss, reg_loss, loss))
#
#                if loss + es_min_delta < best_loss:
#                    best_loss = loss
#                    best_epoch = epoch
#
#                    save_checkpoint_continual_learning(model, f'efficientdet-d{compound_coef}_{epoch}.pth', os.path.join(saved_path, project_name))
#                    best_weights = f'continual-learning/weights/efficientdet-d{compound_coef}_min.pth'
#                model.train()
#
#                # Early stopping
#                if epoch - best_epoch > es_patience > 0:
#                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
#
#                    break
    except KeyboardInterrupt:
        save_checkpoint_continual_learning(model, f'efficientdet-d{compound_coef}_{epoch}.pth', os.path.join(saved_path, project_name, model_name))

    save_checkpoint_continual_learning(model, f'efficientdet-d{compound_coef}_{epoch}.pth', os.path.join(saved_path, project_name, model_name))
    if len(os.listdir(os.path.join(saved_path, project_name, model_name))) == 0:
        print('NO WEIGHTS SAVED')
        best_weights = weights_path
    else: 
        best_weights = get_last_weights(os.path.join(saved_path, project_name, model_name))
    print('best weights ', best_weights)
    return best_weights


def run_efficientdet(orientation_to_file,  model, gpu, car_thresh, person_thresh):

    threshold = 0.05
    use_cuda = gpu >= 0
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    orientation_to_car_count = {}
    orientation_to_person_count = {}

    orientation_to_cars_detected = {}
    orientation_to_people_detected = {}
    # In format frame,orientation,file
    with torch.no_grad():
        for o in orientation_to_file:
            image_path  = orientation_to_file[o]
#            print('Processing ', image_path)
            ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef], mean=model_params.mean, std=model_params.std)
            x = torch.from_numpy(framed_imgs[0])

            if use_cuda:
                x = x.cuda(gpu)
                if use_float16:
                    x = x.half()
                else:
                    x = x.float()
            else:
                x = x.float()

            people_count = 0
            car_count = 0
            x = x.unsqueeze(0).permute(0, 3, 1, 2)
            features, regression, classification, anchors = model(x)

            preds = postprocess(x,
                                anchors, regression, classification,
                                regressBoxes, clipBoxes,
                                threshold, nms_threshold)
            if not preds:
                orientation_to_car_count[o] = 0
                orientation_to_person_count[o] = 0
                continue

            preds = invert_affine(framed_metas, preds)[0]

            scores = preds['scores']
            class_ids = preds['class_ids']
            rois = preds['rois']
            idx = 0

            orientation_to_cars_detected[o] = []
            orientation_to_people_detected[o] = []
            if len(scores) == 0:
                orientation_to_car_count[o] = 0
                orientation_to_person_count[o] = 0
                continue
            for i in range(0,len(scores)):
                # Class id 0 is car, 1 is person
                if scores[i] >= car_thresh and class_ids[i] == 0 :
                    car_count += 1
                    orientation_to_cars_detected[o].append(  rois[i])
                elif scores[i] >= person_thresh  and class_ids[i] == 1:
#                    print(type(rois[i]))
#                    print( rois[i][0]  )
                    orientation_to_people_detected[o].append(rois[i])
                    people_count += 1
            orientation_to_car_count[o] = car_count
            orientation_to_person_count[o] = people_count
    return orientation_to_car_count, orientation_to_person_count, orientation_to_cars_detected, orientation_to_people_detected



def find_directions_from_orientations(worst_orientation, best_orientation, orientations):
    worst_pan = extract_pan(worst_orientation)
    best_pan = extract_pan(best_orientation)

    worst_tilt = extract_tilt(worst_orientation)
    best_tilt = extract_tilt(best_orientation)
   
    best_is_left = False
    best_is_right = False
    best_is_top = False
    best_is_bottom = False 
    if worst_pan != best_pan:
        while worst_pan != best_pan:
            # Keep going left. If we reach best pan, we know best pan is to the left.
            # Otherwise, we run into the end of the region and know best pan is to the right
            new_worst_pan = rotate_left(worst_pan)
            if new_worst_pan == worst_pan:
                best_is_right = True
                break
            worst_pan = new_worst_pan
        if not best_is_right:
            best_is_left = True
    if worst_tilt != best_tilt:
        if best_tilt > worst_tilt:
            best_is_top = True
        else:
            best_is_bottom = True
    return best_is_left, best_is_top, best_is_right, best_is_bottom




def neighboring_orientations_madeye(anchor_orientation, current_formation, orientations, orientation_to_historical_scores):
    total_num_historical_scores = 0
    for o in orientation_to_historical_scores:
        total_num_historical_scores += orientation_to_historical_scores[o]

    if len(current_formation) == 0 or total_num_historical_scores < 7:
        is_left = False
        is_right = False
        is_top = False
        is_bottom = False
        o1 = rotate_left(anchor_orientation, orientations)
        if o1 == anchor_orientation:
            is_left = True 
        o2 = rotate_right(anchor_orientation, orientations)
        if o2 == anchor_orientation:
            is_right = True
        o3 = rotate_up(anchor_orientation, orientations)
        if o3 == anchor_orientation:
            is_top = True
        o4 = rotate_down(anchor_orientation, orientations)
        if o4 == anchor_orientation:
            is_bottom = True
        # Generate alternate orientations if we're on the edge of the region
        if is_left:
            # If anchor is far left, o1 is bad
            if is_top:
                # anchor is in top left, o3 is bad too
                o1 = rotate_down(o2, orientations)
                o3 =  rotate_down(o1, orientations)
            elif is_bottom:
                # anchor is in bottom left, o4 is bad too
                o1 = rotate_up(o2, orientations)
                o4 = rotate_up(o1, orientations)
            else:
                # Top right by default
                o1 = rotate_up(o2, orientations)

        if is_right:
            # If current orientation is far right, o2 is bad
            if is_top:
                # anchor is top right, o3 is bad too
                o2 = rotate_down(o1, orientations)
                o3 = rotate_down(o2, orientations)
            elif is_bottom:
                # anchor is bottom right, o4 is bad too
                o2 = rotate_up(o1, orientations)
                o4 = rotate_up(o2, orientations)
            else:
                # Top left by default
                o2 = rotate_up(o1, orientations)

        if is_top:
            # If current orientation is top, o3 is bad
            if is_left:
                # anchor is far left, o1 is bad
                o1 = rotate_down(o2, orientations)
                o3 = rotate_down(o1, orientations)
            elif is_right:
                # anchor is far right, o2 is bad
                o2 = rotate_down(o1, orientations)
                o3 = rotate_down(o2, orientations)
            else:
                # Bottom right by default
                o3 = rotate_down(o2, orientations)
        if is_bottom:
            # If current orientation is bottom, o4 is bad
            if is_left:
                # anchor is far left, o1 is bad
                o1 = rotate_up(o2, orientations)
                o4 = rotate_up(o1, orientations)
            elif is_right:
                # anchor is far right, o2 is bad
                o2 = rotate_up(o1, orientations)
                o4 = rotate_up(o2, orientations)
            else:
                # Top left by default
                o4 = rotate_up(o1, orientations)
        return [o1, o2, o3, o4, anchor_orientation]
    # Swap out worst orientation with a new orientation in the direction of best
    orientation_to_avg_count = {}
    worst_orientation = current_formation[0]
    worst_score = 1000.0
    best_orientation = current_formation[0]
    best_score = -1000.0
    for o in orientation_to_historical_scores:
        orientation_to_avg_count[o] = sum(orientation_to_historical_scores[o]) / len(orientation_to_historical_scores[o])
        if orientation_to_avg_count[o] < worst_score:
            worst_score = orientation_to_avg_count[o]
            worst_orientation = o
        if orientation_to_avg_count[o] > best_score:
            best_score = orientation_to_avg_count[o]
            best_orientation = o
    if worst_score == anchor_orientation:
        orientation_to_historical_scores.clear()
        return current_formation
    if best_score / worst_score >= 2.0:
        # Try to get new orientation in direction of best direction
        best_is_left, best_is_top, best_is_right, best_is_bottom = find_directions_from_orientations(worst_orientation, best_orientation, orientations)
        best_orientation_copy = best_orientation
        new_orientation = best_orientation
        if best_is_left:
            new_orientation = rotate_left(best_orientation_copy)
            if best_is_top or (new_orientation == best_orientation_copy or new_orientation in current_formation):
                new_orientation = rotate_up(best_orientation_copy)
            elif best_is_bottom or  (new_orientation == best_orientation_copy or new_orientation in current_formation):
                new_orientation = rotate_down(best_orientation_copy)
        elif best_is_right:
            new_orientation = rotate_right(new_orientation)
            if best_is_top or (new_orientation == best_orientation_copy  or new_orientation in current_formation):
                new_orientation = rotate_up(best_orientation_copy)
            elif best_is_bottom or  (new_orientation == best_orientation_copy or new_orientation in current_formation):
                new_orientation = rotate_down(best_orientation_copy)
        if best_is_top and (new_orientation == best_orientation_copy or new_orientation in current_formation):
            new_orientation = rotate_top(best_orientation_copy)
        elif best_is_bottom and (new_orientation == best_orientation_copy or new_orientation in current_formation):
            new_orientation = rotate_bottom(best_orientation_copy)
        if new_orientation == best_orientation_copy or new_orientation in current_formation:
            # best orientation is already at edge of region or new orientation is already in the current formation,
            # return old formation
            return current_formation
        new_formation = []
        for o in current_formation:
            if o == worst_orientation:
                continue 
            new_formation.append(o)
        new_formation.append(new_orientation)
        return new_formation
    else:
        return current_formation



def zoom_out(current_orientation, orientations):
    pan = extract_pan(current_orientation)
    tilt = extract_tilt(current_orientation)
    zoom = extract_zoom(current_orientation)
    if zoom > 1:
        zoom -= 1
    new_orientation = "{}-{}-{}".format(pan, tilt, zoom)
    return new_orientation

def zoom_in(current_orientation, orientations):
    pan = extract_pan(current_orientation)
    tilt = extract_tilt(current_orientation)
    zoom = extract_zoom(current_orientation)
    if zoom < 3:
        zoom += 1
    new_orientation = "{}-{}-{}".format(pan, tilt, zoom)
    return new_orientation

def rotate_up(current_orientation, orientations):
    pan = extract_pan(current_orientation)
    tilt = extract_tilt(current_orientation)
    zoom = extract_zoom(current_orientation)
    if tilt < 30:
        tilt += 15
    new_orientation = "{}-{}-{}".format(pan, tilt, zoom)
    if new_orientation in orientations:
        return new_orientation
    else:
        return current_orientation


def rotate_down(current_orientation, orientations):
    pan = extract_pan(current_orientation)
    tilt = extract_tilt(current_orientation)
    zoom = extract_zoom(current_orientation)
    if tilt > -30:
        tilt -= 15
    new_orientation = "{}-{}-{}".format(pan, tilt, zoom)
    if new_orientation in orientations:
        return new_orientation
    else:
        return current_orientation

def rotate_left(current_orientation, orientations):
    pan = extract_pan(current_orientation)
    tilt = extract_tilt(current_orientation)
    zoom = extract_zoom(current_orientation)
    pan -= 30
    if pan < 0:
        pan = 330
    new_orientation = "{}-{}-{}".format(pan, tilt, zoom)
    if new_orientation in orientations:
        return new_orientation
    else:
        return current_orientation

def rotate_right(current_orientation, orientations):
    pan = extract_pan(current_orientation)
    tilt = extract_tilt(current_orientation)
    zoom = extract_zoom(current_orientation)
    pan += 30
    if pan > 330:
        pan = 0
    new_orientation = "{}-{}-{}".format(pan, tilt, zoom)
    if new_orientation in orientations:
        return new_orientation
    else:
        return current_orientation
 

def find_best_dynamic_score(workload,
                       frame_begin,
                       frame_limit,
                       orientations,
                       frame_to_model_to_orientation_to_car_count,
                       frame_to_model_to_orientation_to_person_count,
                       frame_to_model_to_orientation_to_car_map,
                       frame_to_model_to_orientation_to_person_map,
                       frame_to_model_to_orientation_to_object_ids,
                       gt_model_to_object_ids
                       ):
   
    total_accuracy = 0.0
    duration_list = []
    current_duration = 0
    num_frames = 0
    num_frames_in_best_fixed = 0
    prev_best_orientation = orientations[0]
    print('gt_model ', gt_model_to_object_ids)

    num_aggregate_queries = evaluation_tools.num_aggregate_queries_in_workload(workload)
    num_non_aggregate_queries = len(workload) - num_aggregate_queries
    print('Num aggregate queries ', num_aggregate_queries)
    print('num non aggregate queries = ', num_non_aggregate_queries)
    print('workload len ', len(workload))
    non_aggregate_accuracy = 0.0
    aggregate_accuracy = 0.0

    # Number of times an orientation was best
    orientation_occurrences = []

    model_to_object_ids_found = {}

    # Scale down length of aggregate groudn truth so per-frame est accuracies aren't too low
    print('BEST DYNAMIC')
    for f in range(frame_begin, frame_limit):
        if f % SKIP != 0:
            continue

        best_fixed_est_current_total_accuracy = 0.0

        max_est_current_total_accuracy = 0.0
        max_current_non_aggregate_accuracy = 0.0
        max_est_current_aggregate_accuracy = 0.0
        best_orientation = orientations[0]

        best_orientations_found = []
        for o in orientations:
            # Go through best fixed orientations to determine max out of them for current frame
            current_non_aggregate_accuracy = compute_accuracy(workload, f, o, orientations,frame_to_model_to_orientation_to_car_count,
                frame_to_model_to_orientation_to_person_count,frame_to_model_to_orientation_to_car_map,frame_to_model_to_orientation_to_person_map)
            aggregate_accuracy = evaluation_tools.compute_aggregate_accuracy(workload, f, frame_limit, o, orientations, model_to_object_ids_found, frame_to_model_to_orientation_to_object_ids)

            est_current_total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + (num_non_aggregate_queries / len(workload)) * current_non_aggregate_accuracy

            if best_fixed_orientation is not None and o == best_fixed_orientation:
                best_fixed_est_current_total_accuracy = est_current_total_accuracy

            if est_current_total_accuracy > max_est_current_total_accuracy:
                max_est_current_total_accuracy = est_current_total_accuracy
                max_current_non_aggregate_accuracy = current_non_aggregate_accuracy
                max_est_current_aggregate_accuracy = aggregate_accuracy
                best_orientation = o
                best_orientations_found = [o]

            elif est_current_total_accuracy == max_est_current_total_accuracy:
                best_orientations_found.append(o)

        best_orientation = select_closest_orientation(prev_best_orientation, best_orientations_found)
            # Update for paper results on ranking and to see discrepancy between best dynamic and others
        orientation_occurrences.append(best_orientation)
 
        find_aggregate_ids_for_frame_and_orientation(workload, f, best_orientation, frame_to_model_to_orientation_to_object_ids, model_to_object_ids_found)
        non_aggregate_accuracy += max_current_non_aggregate_accuracy

        # compute percentage of frames spent in best fixed
        if best_fixed_orientation is not None:
            if best_fixed_est_current_total_accuracy  == max_est_current_total_accuracy:
                num_frames_in_best_fixed += 1
        # Keep track of durations of best orientation
        current_duration += 1
        if prev_best_orientation != best_orientation:
            duration_list.append(current_duration)
            current_duration = 0
        prev_best_orientation = best_orientation
        num_frames += 1
    non_aggregate_accuracy /= num_frames
    aggregate_accuracy = evaluation_tools.evaluate_aggregate_queries(workload, frame_begin, frame_limit, orientations, model_to_object_ids_found, 
                           frame_to_model_to_orientation_to_object_ids, gt_model_to_object_ids)

    total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + (num_non_aggregate_queries / len(workload)) * non_aggregate_accuracy
    return total_accuracy 




def oracle_select_orientation(workload,
                    current_frame,
                    frame_limit,
                    current_formation,
                    orientations,
                    model_to_object_ids_found,
                    frame_to_model_to_orientation_to_car_count,
                    frame_to_model_to_orientation_to_person_count,
                    frame_to_model_to_orientation_to_car_map,
                    frame_to_model_to_orientation_to_person_map, 
                    frame_to_model_to_orientation_to_object_ids,
                    gt_model_to_object_ids):
    num_frames = 0
    num_aggregate_queries = evaluation_tools.num_aggregate_queries_in_workload(workload)
    best_accuracy = -1.0
    best_orientation = orientations[0]
    for o in current_formation:
        model_to_object_ids_found = {}
        non_aggregate_accuracy = evaluation_tools.compute_accuracy(workload, current_frame, o, orientations,frame_to_model_to_orientation_to_car_count,
            frame_to_model_to_orientation_to_person_count,frame_to_model_to_orientation_to_car_map,frame_to_model_to_orientation_to_person_map)
        current_model_to_object_ids_found = {}
        evaluation_tools.find_aggregate_ids_for_frame_and_orientation(workload, current_frame, o, frame_to_model_to_orientation_to_object_ids, current_model_to_object_ids_found)
        aggregate_accuracy = evaluation_tools.compute_aggregate_accuracy(workload, current_frame, frame_limit, o, orientations, model_to_object_ids_found, frame_to_model_to_orientation_to_object_ids)
        total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + ((len(workload) - num_aggregate_queries) / len(workload)) * non_aggregate_accuracy
        if total_accuracy > best_accuracy:
            best_accuracy = total_accuracy
            best_orientation = o 
    return best_orientation

def compute_shortest_distance(orientation, shape_orientations, all_orientations):
    orientation = orientation[:-1]+'1'
    shape_orientations = [s[:-1]+'1' for s in shape_orientations]
    all_orientations = [s[:-1]+'1' for s in all_orientations]
    all_orientations = set(all_orientations)
    shape_orientations = set(shape_orientations)
    bfs_queue = [(orientation,0)]
    visited = set()
    while len(bfs_queue) > 0:
        (current_orientation, gap) = bfs_queue.pop(0)
        if current_orientation in shape_orientations:
            return gap
        neighbors = get_all_neighbors(current_orientation)
        for n in neighbors:
            if n not in visited:
                visited.add(n)
                if n in all_orientations:
                    bfs_queue.append((n, gap + 1))
            
    return 100000

def print_stats_about_our_perf(trace_of_our_orientations, trace_of_static_cross_formation, trace_of_best_dynamic_orientations, trace_of_distance_between_us_and_best_dynamic, trace_of_regions_we_chose, static_cross_formation):
    print(f"trace of our111 orientations: {trace_of_our_orientations}")
    print(f"trace of static orientations: {trace_of_static_cross_formation}")
    print(f"trace of dynami orientations: {trace_of_best_dynamic_orientations}")
    print(f"trace of dynamic distances  : {trace_of_distance_between_us_and_best_dynamic}")
    backup_trace_of_best_dynamic_orientations = trace_of_best_dynamic_orientations
    backup_trace_of_regions_we_chose = trace_of_regions_we_chose
    backup_static_cross_formation = static_cross_formation
    # input(f'one set of frames complete')
    number_of_indices_that_overlap_ignoring_zoom = 0
    number_of_indices_that_overlap_considering_zoom = 0
    for i in range(len(trace_of_our_orientations)):
        if trace_of_best_dynamic_orientations[i] in set(trace_of_regions_we_chose[i]["our_region"]):
            number_of_indices_that_overlap_considering_zoom += 1
        trace_of_best_dynamic_orientations[i] = trace_of_best_dynamic_orientations[i][:-1]+'1' 
        trace_of_regions_we_chose[i]["our_region"] = [x[:-1] + '1' for x in trace_of_regions_we_chose[i]["our_region"]]
        if trace_of_best_dynamic_orientations[i] in set(trace_of_regions_we_chose[i]['our_region']):
            number_of_indices_that_overlap_ignoring_zoom += 1
    print(f"mur: ours: num choices we made: {len(trace_of_our_orientations)}. out of that, fraction of chosen orientations that overlap with zoom: {number_of_indices_that_overlap_considering_zoom/len(trace_of_our_orientations)}, ignoring zoom: {number_of_indices_that_overlap_ignoring_zoom/len(trace_of_our_orientations)}")
    with open("our_results.json", "a") as f:
        f.write(f"mur: ours: num choices we made: {len(trace_of_our_orientations)}. out of that, fraction of chosen orientations that overlap with zoom: {number_of_indices_that_overlap_considering_zoom/len(trace_of_our_orientations)}, ignoring zoom: {number_of_indices_that_overlap_ignoring_zoom/len(trace_of_our_orientations)}\n")
    trace_of_best_dynamic_orientations = backup_trace_of_best_dynamic_orientations
    number_of_indices_that_overlap_ignoring_zoom = 0
    number_of_indices_that_overlap_considering_zoom = 0
    for i in range(len(trace_of_static_cross_formation)):
        # print(f"trace_of_best_dynamic_orientations[i]:{trace_of_best_dynamic_orientations[i]}")
        # print(f"trace_of_static_cross_formation[i]:{trace_of_static_cross_formation[i]}")
        if trace_of_best_dynamic_orientations[i] in set(static_cross_formation):
            number_of_indices_that_overlap_considering_zoom += 1
            # print(f"number_of_indices_that_overlap_considering_zoom+=1")
        trace_of_best_dynamic_orientations[i] = trace_of_best_dynamic_orientations[i][:-1]+'1' 
        static_cross_formation = [x[:-1] + '1' for x in static_cross_formation]
        if trace_of_best_dynamic_orientations[i] in set(static_cross_formation):
            number_of_indices_that_overlap_ignoring_zoom += 1
    #         print(f"number_of_indices_that_overlap_ignoring_zoom+=1")
    # print(f"number_of_indices_that_overlap_ignoring_zoom is {number_of_indices_that_overlap_ignoring_zoom}")
    # print(f"number_of_indices_that_overlap_considering_zoom is {number_of_indices_that_overlap_considering_zoom}")
    print(f"mur: static: num choices we made: {len(trace_of_our_orientations)}. out of that, fraction of chosen orientations that overlap with zoom: {number_of_indices_that_overlap_considering_zoom/len(trace_of_our_orientations)}, ignoring zoom: {number_of_indices_that_overlap_ignoring_zoom/len(trace_of_our_orientations)}")
    with open("our_results.json", "a") as f:
        # f.write(f"mur: static choices: {trace_of_static_cross_formation}")
        # f.write(f"mur: dynamic choices: {trace_of_best_dynamic_orientations}")
        f.write(f"mur: static: num choices we made: {len(trace_of_our_orientations)}. out of that, fraction of chosen orientations that overlap with zoom: {number_of_indices_that_overlap_considering_zoom/len(trace_of_our_orientations)}, ignoring zoom: {number_of_indices_that_overlap_ignoring_zoom/len(trace_of_our_orientations)}\n")
    with open("our_results.json", "a") as f:
        f.write(f"mur: choices: ours: {trace_of_our_orientations}, static: {trace_of_static_cross_formation}, dynamic: {trace_of_best_dynamic_orientations}")
    trace_of_regions_we_chose = backup_trace_of_regions_we_chose
    static_cross_formation = backup_static_cross_formation

def run_madeye(name,
              inference_dir,
               rectlinear_dir,
               params,
               workload,
               frame_begin,
               frame_limit,
               orientations,
               anchor_orientation,
               frame_to_model_to_orientation_to_car_count,
               frame_to_model_to_orientation_to_person_count,
               frame_to_model_to_orientation_to_cars_detected,
               frame_to_model_to_orientation_to_people_detected,
               frame_to_model_to_orientation_to_car_map,
               frame_to_model_to_orientation_to_person_map,
               frame_to_model_to_orientation_to_object_ids,
              frame_to_model_to_orientation_to_efficientdet_cars_detected,
              frame_to_model_to_orientation_to_efficientdet_people_detected,
               gt_model_to_object_ids,
                num_frames_to_send,
               num_frames_to_keep,
                blacklisted_frames=[]
               ):

    # ** EfficientDet initialization **

    model_to_car_thresh = {}
    model_to_person_thresh = {}
    model_to_car_thresh['faster-rcnn'] = 0.3
    model_to_car_thresh['yolov4'] = 0.3
    model_to_car_thresh['tiny-yolov4'] = 0.3
    model_to_car_thresh['ssd-voc'] = 0.3

    model_to_person_thresh['faster-rcnn'] = 0.2
    model_to_person_thresh['yolov4'] = 0.2
    model_to_person_thresh['tiny-yolov4'] = 0.2
    model_to_person_thresh['ssd-voc'] = 0.2
    data_path = 'continual-learning-temp/datasets/'
    saved_path = 'continual-learning-temp/weights/'

    # Remove old weights 
    if os.path.exists(saved_path):
        shutil.rmtree(saved_path)

    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(saved_path, exist_ok=True)

    model_to_weights_paths = {}
    model_to_weights_paths['faster-rcnn'] = params['faster_rcnn_weights']
    model_to_weights_paths['yolov4'] = params['yolov4_weights']
    model_to_weights_paths['tiny-yolov4'] = params['tiny_yolov4_weights']
    model_to_weights_paths['ssd-voc'] = params['ssd_voc_weights']
    gpu = params['gpu']
    weights_path = None
    model_to_efficientdet = {}
    for q in workload:
        if params['use_efficientdet']:
            efficientdet = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=eval(model_params.anchors_ratios), scales=eval(model_params.anchors_scales))
            if q[0] in model_to_weights_paths:
                weights_path = model_to_weights_paths[q[0]]
                print('Loading ', weights_path)
                if gpu >= 0:
                    efficientdet.load_state_dict(torch.load(weights_path))
                    efficientdet.to(f'cuda:{gpu}')
                else:
                    efficientdet.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            efficientdet.requires_grad_(False)
            model_to_efficientdet[q[0]] = efficientdet
        else:
            model_to_efficientdet[q[0]] = None

    orientation_to_training_frames = {}
    orientation_to_val_frames = {}


    orientation_to_visits = {}
    gt_orientation_to_visits = {}

    # ** End EfficientDet stuff **
    orientation_to_frames_since_last_visit = {}
    num_aggregate_queries = evaluation_tools.num_aggregate_queries_in_workload(workload)


    running_non_aggregate_accuracy = 0.0
    running_gt_non_aggregate_accuracy = 0.0


    model_to_object_ids_found = {}
    best_gt_model_to_object_ids_found = {}
    orientation_to_historical_scores = {}
    orientation_to_historical_counts = {}

    car_query_weight = evaluation_tools.num_car_queries_in_workload(workload) / len(workload)
    person_query_weight = 1.0 - car_query_weight
    num_frames = 0
    print('Best fixed orientation ', anchor_orientation)

    with open('trace.txt', 'a') as f_trace:
        f_trace.write(f'{frame_begin} -- {frame_limit}\n' )
        f_trace.write(str(workload) + '\n')
        f_trace.write(anchor_orientation + '\n')

    current_formation = []
    static_cross_formation = [anchor_orientation, 
                                rotate_down(anchor_orientation, orientations),
                                rotate_left(anchor_orientation, orientations),
                                rotate_right(anchor_orientation, orientations),
                                rotate_up(anchor_orientation, orientations)]
    current_formation = static_cross_formation
    trace_of_our_orientations = []
    trace_of_best_dynamic_orientations = []
    trace_of_static_cross_formation = []
    trace_of_regions_we_chose = []
    trace_of_distance_between_us_and_best_dynamic = []
    orientation_to_visited_step_numbers = {}
    peek_orientations = set()
    step_num = 1
    zoom_explorations_in_progress = {}

    for f in range(frame_begin, frame_begin + int(0.3*(frame_limit - frame_begin))):
        if f % SKIP * 2 != 0:
            continue
        for o in orientations:
            orientation_to_frames_since_last_visit[o] = 0
            if o not in orientation_to_training_frames:
                orientation_to_training_frames[o] = []
            orientation_to_training_frames[o].append(f)
            if f % SKIP * 8 == 0:
                if extract_zoom(o) == 2 and random.random() <= 0.5:
                    continue
                if extract_zoom(o) == 3 and random.random() <= 0.75:
                    continue
                if o not in orientation_to_val_frames:
                    orientation_to_val_frames[o] = []
                orientation_to_val_frames[o].append(f)

    ranks = []
    for f in range(frame_begin + int(0.3*(frame_limit - frame_begin)), frame_limit + 1):
        if f % SKIP != 0:
            continue
        if f in blacklisted_frames:
            continue
        print(f"\nframe: {f}")
        if f not in frame_to_model_to_orientation_to_efficientdet_cars_detected:
            frame_to_model_to_orientation_to_efficientdet_cars_detected[f] = {}
        if f not in frame_to_model_to_orientation_to_efficientdet_people_detected:
            frame_to_model_to_orientation_to_efficientdet_people_detected[f] = {}

        when_was_an_orientation_seen_last = evaluation_tools.compute_when_an_orientation_was_last_visited(trace_of_our_orientations, current_formation)

        
        orientation_to_current_scores = {}
        orientation_to_current_counts = {}
        orientation_to_current_mike_factor = {}
        # ** EfficientDet inference **

        orientation_to_file = {} 
        for o in current_formation:
            orientation_to_file[o] = os.path.join(rectlinear_dir, o, f'frame{f}.jpg')
        model_to_orientation_to_efficientdet_car_count = {}
        model_to_orientation_to_efficientdet_person_count = {}
        model_to_orientation_to_efficientdet_cars_detected = {}
        model_to_orientation_to_efficientdet_people_detected = {}
        for m in model_to_efficientdet:
            model_to_orientation_to_efficientdet_car_count[m] = {}
            model_to_orientation_to_efficientdet_person_count[m] = {}

            model_to_orientation_to_efficientdet_cars_detected[m] = {}
            model_to_orientation_to_efficientdet_people_detected[m] = {}
            gt_orientation_to_car_count = {}
            gt_orientation_to_person_count = {}

            gt_orientation_to_cars_detected = {}
            gt_orientation_to_people_detected = {}
            for o in orientation_to_file:
                gt_orientation_to_car_count[o] = frame_to_model_to_orientation_to_car_count[f][m][o]
                gt_orientation_to_person_count[o] = frame_to_model_to_orientation_to_person_count[f][m][o]
                detected = []
                for obj in frame_to_model_to_orientation_to_cars_detected[f][m][o]:
                    detected.append([obj.left, obj.top, obj.right, obj.bottom]) 
                gt_orientation_to_cars_detected[o] = detected 

                detected = []
                for obj in frame_to_model_to_orientation_to_people_detected[f][m][o]:
                    detected.append([obj.left, obj.top, obj.right, obj.bottom]) 
                gt_orientation_to_people_detected[o] = detected
            if params['use_efficientdet']:
                orientation_to_efficientdet_car_count, orientation_to_efficientdet_person_count, orientation_to_efficientdet_cars_detected, orientation_to_efficientdet_people_detected = run_efficientdet(orientation_to_file,  model_to_efficientdet[m], gpu, model_to_car_thresh[m], model_to_person_thresh[m])
            else:
                orientation_to_efficientdet_car_count = gt_orientation_to_car_count
                orientation_to_efficientdet_person_count = gt_orientation_to_person_count
                orientation_to_efficientdet_cars_detected = gt_orientation_to_cars_detected
                orientation_to_efficientdet_people_detected = gt_orientation_to_people_detected

            model_to_orientation_to_efficientdet_car_count[m] = orientation_to_efficientdet_car_count
            model_to_orientation_to_efficientdet_person_count[m]= orientation_to_efficientdet_person_count

            model_to_orientation_to_efficientdet_cars_detected[m] = orientation_to_efficientdet_cars_detected
            model_to_orientation_to_efficientdet_people_detected[m]= orientation_to_efficientdet_people_detected

#            orientation_to_efficientdet_cars_detected , orientation_to_efficientdet_people_detected = run_efficientdet(orientation_to_file,  model_to_efficientdet[m], gpu, model_to_car_thresh[m], model_to_person_thresh[m])
#            print('MODEL ', m)
#
#            gt_orientation_to_car_count = {}
#            gt_orientation_to_person_count = {}
#            for o in orientation_to_file:
#                model_to_orientation_to_efficientdet_car_count[m][o] = len(orientation_to_efficientdet_cars_detected)
#                model_to_orientation_to_efficientdet_person_count[m][o] = len(orientation_to_efficientdet_people_detected)
#                gt_orientation_to_car_count[o] = frame_to_model_to_orientation_to_car_count[f][m][o]
#                gt_orientation_to_person_count[o] = frame_to_model_to_orientation_to_person_count[f][m][o]
#
#            print("EFFICIENTDET")
#            print('car count ', model_to_orientation_to_efficientdet_car_count[m])
#            print('ppl count ', model_to_orientation_to_efficientdet_person_count[m])
#
#            print("GT COUNTS")
#            print('car count ', gt_orientation_to_car_count)
#            print('ppl count ', gt_orientation_to_person_count)
            


        orientation_to_score = {}
        print('Current formation ', current_formation)
        with open('trace.txt', 'a') as f_trace:
            f_trace.write(str(current_formation)
                          .replace('\'', '').replace('[', '').replace(']', '') + '\n')
        for o in current_formation:

            score = evaluation_tools.get_mikes_mike_factor(workload, 
                                 f,
                                 o,
                                 current_formation,
                                 model_to_orientation_to_efficientdet_car_count,
                                 model_to_orientation_to_efficientdet_person_count,
                                 model_to_orientation_to_efficientdet_cars_detected,
                                 model_to_orientation_to_efficientdet_people_detected,
                                 orientation_to_frames_since_last_visit,
                                orientation_to_visits)
            orientation_to_score[o] = score
            orientation_to_current_counts[o] = evaluation_tools.get_count_of_orientation(workload, f, o, orientations,frame_to_model_to_orientation_to_car_count,
                frame_to_model_to_orientation_to_person_count,frame_to_model_to_orientation_to_car_map,frame_to_model_to_orientation_to_person_map)

#        print('Raw counts')
#         
#        for m in model_to_orientation_to_efficientdet_car_count:
#            print('**' , m, ' **')
#            print('\torientation to car counts ', model_to_orientation_to_efficientdet_car_count[m])
#            print('\torientation to ppl counts ', model_to_orientation_to_efficientdet_person_count[m])
#        print('orientation to scores ', orientation_to_score)
        current_orientation_to_ranking = rank_orientations(orientation_to_score)
#        print('Current orientation to rank ', current_orientation_to_ranking)
        # ** End EffientDet stuff **

        ### GET ground truth values for eval

        orientation_to_actual_est_accuracies = {}
        for o in current_formation:
            nax = evaluation_tools.compute_accuracy(workload, f, o, orientations,frame_to_model_to_orientation_to_car_count,
                frame_to_model_to_orientation_to_person_count,frame_to_model_to_orientation_to_car_map,frame_to_model_to_orientation_to_person_map)

            aax = evaluation_tools.compute_aggregate_accuracy(workload, f, frame_limit, o, orientations, best_gt_model_to_object_ids_found, frame_to_model_to_orientation_to_object_ids )

            total_ax = (num_aggregate_queries / len(workload)) * aax + ((len(workload) - num_aggregate_queries) / len(workload)) * nax
            orientation_to_actual_est_accuracies[o] = total_ax

        sorted_orientations = []
        sorted_dict = {k: v for k, v in sorted(current_orientation_to_ranking.items(), key=lambda item: item[1] )}
        best_orientations = []
        for o in sorted_dict:
            best_orientations.append(o)
        min_visits = 100000
        current_orientation = best_orientations[0]
#        for o in best_orientations:
#            if o not in orientation_to_visits:
#                orientation_to_visits[o] = 0
#            if orientation_to_visits[o] < min_visits:
#                current_orientation = o
#                min_visits = orientation_to_visits[o]



#        if random.random() <= 0.2:
#            if len(sorted_orientations) > 5:
#                current_orientation = random.choice(sorted_orientations[:4])
#            else:
#                current_orientation = random.choice(sorted_orientations)
        print('Actual est accuracies ', orientation_to_actual_est_accuracies)
        orientation_to_gt_ranking = rank_orientations(orientation_to_actual_est_accuracies)





        if current_orientation  in orientation_to_gt_ranking:
            ranks.append(orientation_to_gt_ranking[current_orientation])
        print('Current orientation ', current_orientation, ' Best fixed is ', anchor_orientation)
        print('Ranking ', ranks[-1])
        print('Orientation to GT rank ', orientation_to_gt_ranking)


        best_current_orientations = []
        gt_sorted_dict = {k: v for k, v in sorted(orientation_to_gt_ranking.items(), key=lambda item: item[1] )}
        for o in gt_sorted_dict:
            if len(best_current_orientations) == 0:
                best_current_orientations.append(o)
            elif sorted_dict[o] == sorted_dict[best_current_orientations[-1]]:
                best_current_orientations.append(o)
            else:
                break
        best_gt_orientation = best_current_orientations[0]
        ### GOt ground truth ####

        # populate the scores for the current frame for each of the orientations


        # Update stats to study freqency of orientation vistis
        if current_orientation not in orientation_to_visits:
            orientation_to_visits[current_orientation] = 0
        orientation_to_visits[current_orientation] += 1
        if best_gt_orientation  not in gt_orientation_to_visits:
            gt_orientation_to_visits[best_gt_orientation] = 0
        gt_orientation_to_visits[best_gt_orientation] += 1
        if num_frames % 20 == 0:
            print('Orientations selected')
            for o in orientation_to_visits:
                if orientation_to_visits[o] > 0:
                    print('\t', o, ' -> ', orientation_to_visits[o])
            print('\nBest GT orientations ')
            for o in gt_orientation_to_visits:
                print('\t', o, ' -> ', gt_orientation_to_visits[o])

        # *********************


        for o in current_formation:
            orientation_to_current_mike_factor[o] = evaluation_tools.get_muralis_mike_factor(workload, 
                                    f,
                                    o,
                                    when_was_an_orientation_seen_last[o],
                                    model_to_orientation_to_efficientdet_car_count,
                                    model_to_orientation_to_efficientdet_person_count,
                                    model_to_orientation_to_efficientdet_cars_detected,
                                    model_to_orientation_to_efficientdet_people_detected)

        model_to_use = 'faster-rcnn'
        if model_to_use not in model_to_orientation_to_efficientdet_people_detected:
            model_to_use = random.choice(list(model_to_orientation_to_efficientdet_people_detected))
        previous_formation, current_formation, formation_to_use, scores_and_deltas_used, step_num, zoom_explorations_in_progress = neighboring_orientations_delta_method_madeye(anchor_orientation, current_formation, orientations, orientation_to_historical_scores, orientation_to_score, orientation_to_current_counts, orientation_to_historical_counts, orientation_to_current_mike_factor, step_num, orientation_to_visited_step_numbers, peek_orientations, model_to_orientation_to_efficientdet_cars_detected[model_to_use], model_to_orientation_to_efficientdet_people_detected[model_to_use], zoom_explorations_in_progress, num_frames_to_keep )
        # print(f"in madeye zoom explorations is {json.dumps(zoom_explorations_in_progress, indent=2)}")
        # print(f"current formation is {current_formation}")
        # input(f"formation to use is {formation_to_use}")
        for o in orientation_to_score:
            if o not in orientation_to_historical_scores:
                orientation_to_historical_scores[o] = []
            orientation_to_historical_scores[o].append(orientation_to_score[o])
        for o in orientation_to_current_counts:
            if o not in orientation_to_historical_counts:
                orientation_to_historical_counts[o] = []
            orientation_to_historical_counts[o].append(orientation_to_score[o])
        orientation_to_current_counts.clear()

















        # ** Save results for continual learniing; save selecte  orientation/frame **
        if o not in orientation_to_training_frames:
            orientation_to_training_frames[o] = []
        orientation_to_training_frames[o].append(f)

        # *****




        for o in orientation_to_frames_since_last_visit:
            if o == current_orientation:
                orientation_to_frames_since_last_visit[o] = 0
                continue
            orientation_to_frames_since_last_visit[o] += 1

        # #####

        orientation_idx = 0
        non_aggregate_accuracies = []
        while orientation_idx < min(len(best_orientations) , num_frames_to_send):
            # Send images to the server
            current_orientation = best_orientations[orientation_idx]
            print('\tSelected orientation ', current_orientation)
            non_aggregate_accuracy = evaluation_tools.compute_accuracy(workload, f, current_orientation, orientations,frame_to_model_to_orientation_to_car_count,
                frame_to_model_to_orientation_to_person_count,frame_to_model_to_orientation_to_car_map,frame_to_model_to_orientation_to_person_map)
            non_aggregate_accuracies.append(non_aggregate_accuracy)
            evaluation_tools.find_aggregate_ids_for_frame_and_orientation(workload, f, current_orientation, frame_to_model_to_orientation_to_object_ids, model_to_object_ids_found)
            orientation_idx += 1

        current_non_aggregate_accuracy = max(non_aggregate_accuracies)
        running_non_aggregate_accuracy += current_non_aggregate_accuracy
        print('Current non agg accuracy ', current_non_aggregate_accuracy)

        gt_non_aggregate_accuracy = evaluation_tools.compute_accuracy(workload, f, best_gt_orientation , orientations,frame_to_model_to_orientation_to_car_count,
            frame_to_model_to_orientation_to_person_count,frame_to_model_to_orientation_to_car_map,frame_to_model_to_orientation_to_person_map)
        evaluation_tools.find_aggregate_ids_for_frame_and_orientation(workload, f, best_gt_orientation, frame_to_model_to_orientation_to_object_ids, best_gt_model_to_object_ids_found)
        running_gt_non_aggregate_accuracy += gt_non_aggregate_accuracy

        print('GT non agg accuracy', gt_non_aggregate_accuracy)

        num_frames += 1


        trace_of_our_orientations.append(current_orientation)

 

        # Continual learning
        if params['continual_learning'] and num_frames % 90 == 0 and num_frames > 0:
            project_name = 'continual-learning'
            if os.path.exists(f'{saved_path}/{project_name}'):
                shutil.rmtree(f'{saved_path}/{project_name}')
            if os.path.exists(f'{data_path}/{project_name}'):
                shutil.rmtree(f'{data_path}/{project_name}')

            for m in model_to_efficientdet:
                # *** Temporary ***
#                orientation_to_training_frames =  {}
#                orientation_to_val_frames = {}
#                orientation_to_training_frames['0-0-1'] = [6, 12, 18]
#                orientation_to_val_frames['0-0-1'] = [6, 12, 18]
                # *****

    
                # Train set
                generate_dataset(inference_dir, rectlinear_dir, m, f, orientation_to_training_frames, 'train', data_path, project_name)
        #        # Val set
                generate_dataset(inference_dir, rectlinear_dir, m, f, orientation_to_val_frames, 'val', data_path,  project_name)
                weights_path = model_to_weights_paths[m]
                new_weights_path = continual_train(m, weights_path, data_path, saved_path, 'test', gpu, num_epochs=6)
                print('Saved weights ', new_weights_path)
                model_to_efficientdet[m].load_state_dict(torch.load(new_weights_path))
                model_to_efficientdet[m].requires_grad_(False)
            orientation_to_training_frames.clear()

            model_to_car_thresh['faster-rcnn'] = 0.3
            model_to_car_thresh['yolov4'] = 0.3
            model_to_car_thresh['tiny-yolov4'] = 0.3
            model_to_car_thresh['ssd-voc'] = 0.3

            model_to_person_thresh['faster-rcnn'] = 0.2
            model_to_person_thresh['yolov4'] = 0.2
            model_to_person_thresh['tiny-yolov4'] = 0.2
            model_to_person_thresh['ssd-voc'] = 0.2



#    print_stats_about_our_perf(trace_of_our_orientations, trace_of_static_cross_formation, trace_of_best_dynamic_orientations, trace_of_distance_between_us_and_best_dynamic, trace_of_regions_we_chose, static_cross_formation)
#    with open(f"our_region_trace.json", "a") as f:
#        f.write(json.dumps(trace_of_regions_we_chose, indent=2))
#        f.write(",\n")



    gt_non_aggregate_accuracy = running_gt_non_aggregate_accuracy / num_frames
    gt_aggregate_accuracy = evaluation_tools.evaluate_aggregate_queries(workload, frame_begin, frame_limit, orientations, best_gt_model_to_object_ids_found, 
                           frame_to_model_to_orientation_to_object_ids, gt_model_to_object_ids)

    gt_total_accuracy = (num_aggregate_queries / len(workload)) * gt_aggregate_accuracy + ((len(workload) - num_aggregate_queries) / len(workload)) * gt_non_aggregate_accuracy
    
    non_aggregate_accuracy = running_non_aggregate_accuracy  / num_frames
    aggregate_accuracy = evaluation_tools.evaluate_aggregate_queries(workload, frame_begin, frame_limit, orientations, model_to_object_ids_found, 
                           frame_to_model_to_orientation_to_object_ids, gt_model_to_object_ids)


    print('Best gt accuracy ', gt_total_accuracy, ' non agg is ', gt_non_aggregate_accuracy, ' agg is ', gt_aggregate_accuracy)
    print('Avg rank ', sum(ranks) / len(ranks))
    print('non agg accuracy ', non_aggregate_accuracy , ' agg accuracy ', aggregate_accuracy)
    total_accuracy = (num_aggregate_queries / len(workload)) * aggregate_accuracy + ((len(workload) - num_aggregate_queries) / len(workload)) * non_aggregate_accuracy
    return total_accuracy, gt_total_accuracy


def save_checkpoint_continual_learning(model, name, saved_path):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(saved_path, name))

def get_all_neighbors(orientation):
    neighbors_with_rotation = get_all_neighbors_with_rotation(orientation)
    neighbors = [n[0] for n in neighbors_with_rotation]
    return neighbors

##### shape helper stuff below
def get_all_neighbors_with_rotation(orientation):
    neighbors = []
    pan = extract_pan(orientation)
    tilt = extract_tilt(orientation)
    zoom = extract_zoom(orientation)
    if pan == 330:
        left_pan = 300
        right_pan = 0
    elif pan == 0:
        left_pan = 330
        right_pan = 30
    else:
        left_pan = pan - 30
        right_pan = pan + 30
    up_tilt = tilt + 15
    down_tilt = tilt - 15
    if left_pan != -1:
        left_orientation = "{}-{}-{}".format(left_pan, tilt, zoom)
        neighbors.append((left_orientation, 30))
    if right_pan != -1:
        right_orientation = "{}-{}-{}".format(right_pan, tilt, zoom)
        neighbors.append((right_orientation, 30))            
    if up_tilt >= -30 and up_tilt <= 30:
        up_orientation = "{}-{}-{}".format(pan, up_tilt, zoom)
        neighbors.append((up_orientation, 15))
    if down_tilt >= -30 and down_tilt <= 30:
        down_orientation = "{}-{}-{}".format(pan, down_tilt, zoom)
        neighbors.append((down_orientation, 15))

    # diagonals
    if left_pan != -1:
        # upper left
        if up_tilt >= -30 and up_tilt <= 30:
            left_up_orientation = "{}-{}-{}".format(left_pan, up_tilt, zoom)
            neighbors.append((left_up_orientation, 33.5))
        # lower left
        if down_tilt >= -30 and down_tilt <= 30:
            left_down_orientation = "{}-{}-{}".format(left_pan, down_tilt, zoom)
            neighbors.append((left_down_orientation, 33.5))
    
    if right_pan != -1:
        # upper right
        if up_tilt >= -30 and up_tilt <= 30:
            right_up_orientation = "{}-{}-{}".format(right_pan, up_tilt, zoom)
            neighbors.append((right_up_orientation, 33.5))
        # lower right
        if down_tilt >= -30 and down_tilt <= 30:
            right_down_orientation = "{}-{}-{}".format(right_pan, down_tilt, zoom)
            neighbors.append((right_down_orientation, 33.5))
    return neighbors

def get_all_neighbors(orientation):
    neighbors_with_rotation = get_all_neighbors_with_rotation(orientation)
    neighbors = [n[0] for n in neighbors_with_rotation]
    return neighbors



def neighboring_orientations_delta_method_madeye(anchor_orientation, 
                                                 current_formation, 
                                                 orientations, 
                                                 orientation_to_historical_scores, 
                                                 orientation_to_current_scores, 
                                                 orientation_to_current_counts, 
                                                 orientation_to_historical_counts, 
                                                 orientation_to_current_mike_factor, 
                                                 step_number, 
                                                 orientation_to_visited_step_numbers, 
                                                 peek_orientations, 
                                                 orientation_to_current_car_boxes, 
                                                 orientation_to_current_person_boxes, 
                                                 zoom_explorations_in_progress, 
                                                 num_frames_to_keep):
    # print(f"muralis function is happenning")
    # print(f"current_formation: {current_formation}")
    # print(f"orientations: {orientations}")
    # print(f"orientation_to_historical_scores: {orientation_to_historical_scores}")
    # print(f"orientation_to_current_scores: {orientation_to_current_scores}")
    # print(f"orientation_to_historical_counts: {orientation_to_historical_counts}")
    # print(f"orientation_to_current_counts: {orientation_to_current_counts}")
    # input("enter to continue")

    def remove_orientations_from_formation(orientations_to_be_removed, formation):
        output_formation = []
        orientations_to_be_removed = set(orientations_to_be_removed)
        for o in formation:
            if o not in orientations_to_be_removed:
                output_formation.append(o)
        return output_formation
    
    def add_orientations_to_formation(orientations_to_be_added, formation):
        for o in orientations_to_be_added:
            formation.append(o)
        return list(set(formation))

    def distance_from_point_to_line(x1, y1, x2, y2, x3, y3):
        if x1 == x2 and y1 == y2: # distance between two points
            return ((y3 - y1)**2 + (x3 - x1)**2)**0.5
        # print(f"x1 is {x1}, y1 is {y1}")
        # print(f"x2 is {x2}, y2 is {y2}")
        # print(f"x3 is {x3}, y3 is {y3}")
        
        # method 1
        # A is x1, y1
        # B is x2, y2
        # C is x3, y3
        # calculates the shortest distance from a point C to a line defined by two points A and B. The input points A, B, and C are represented by their x and y coordinates.
        # The code first calculates the difference between the x and y coordinates of points A and B to obtain the direction vector of the line. Then it calculates the projection of the vector from point A to point C onto the direction vector of the line, and stores it in the variable u.
        # Next, the code checks if u is outside the range of 0 to 1. If u is greater than 1, it means that the closest point on the line is beyond point B, so the code sets u to 1. If u is less than 0, it means that the closest point on the line is before point A, so the code sets u to 0.
        # Finally, the code calculates the closest point on the line to point C by using u to find the linear combination of the direction vector and point A, and stores the result in variables x and y. 
        # The code then calculates the difference between the x and y coordinates of the closest point on the line and point C, and finds the Euclidean distance between the two points. 
        # The result of this calculation is returned as the shortest distance from point C to the line.
        px = x2 - x1
        py = y2 - y1
        something = px * px + py * py
        # print(f"something is {something}")
        u = ((x3 - x1) * px + (y3 - y1) * py) / float(something)
        if u > 1:
            u = 1
        elif u < 0:
            u = 0
        x = x1 + u * px
        y = y1 + u * py
        dx = x - x3
        dy = y - y3
        # print(f"(dx * dx + dy * dy) is {(dx * dx + dy * dy)}")
        dist = (dx * dx + dy * dy)**0.5
        # input(f"returning {dist}")
        return dist

        # method 2 (simpler)
        # # https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
        # p1 = np.asarray((x1, y1))
        # p2 = np.asarray((x2, y2))
        # p3 = np.asarray((x3, y3))
        # return np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)

    # for the given orientation, get all 8 neighbors
    # and return the list of neighbors that are currently not in the shape
    # but are still within the bounds of the region
    def get_neighbors_outside_shape(orientation, current_formation, all_orientations):
        all_neighbors = get_all_neighbors(orientation)
        current_orientations_set = set(current_formation)
        all_orientations_set = set(all_orientations)
        potential_candidate_neighbors = []

        for neighbor in all_neighbors:
            if neighbor not in current_orientations_set:
                if neighbor in all_orientations_set:
                    potential_candidate_neighbors.append(neighbor)

        return potential_candidate_neighbors
    
    def can_expand_right(num_prior_expansions_to_this_right, lower_score, higher_score):
        if lower_score == 0.0:
            return True
        else:
            ratio = higher_score/lower_score
            if num_prior_expansions_to_this_right == 0:
                return True if ratio >= 1.1 else False
            elif num_prior_expansions_to_this_right == 1: 
                # don't allow expansions on same node repeatedly by setting unrealistic target
                return True if ratio >= 2 else False
            else:
                return True if ratio >= 2.5 else False

    def can_swap_right(num_prior_expansions_to_this_right, lower_score, higher_score):
        if lower_score == 0.0:
            return True
        else:
            ratio = higher_score/lower_score
            if num_prior_expansions_to_this_right == 0:
                return True if ratio >= 1.5 else False
            elif num_prior_expansions_to_this_right == 1:
                return True if ratio >= 1.75 else False
            else:
                return True if ratio >= 2 else False

    def only_added_orientation_last_frame(orientation, orientation_to_visited_step_numbers, current_step_num):
        if orientation not in orientation_to_visited_step_numbers:
            # haven't seen before
            return True
        else:
            if len(orientation_to_visited_step_numbers[o]) == 1:
                # have seen once before so it must be newly added and this is it's second frame in shape
                return True
            else:
                last_visit = orientation_to_visited_step_numbers[o][-1]
                penultimate_visit = orientation_to_visited_step_numbers[o][-2]
                if last_visit - penultimate_visit > 1: # visited last time, but didn't visit before that => newly added last time
                    return True
                else:
                    return False

    def extrapolate_orientation(right_orientation, neighbor_used_for_this_extension, orientations):
        pan_1 = extract_pan(right_orientation)
        tilt_1 = extract_tilt(right_orientation)
        zoom_1 = extract_zoom(right_orientation)

        pan_2 = extract_pan(neighbor_used_for_this_extension)
        tilt_2 = extract_tilt(neighbor_used_for_this_extension)
        zoom_2 = extract_zoom(neighbor_used_for_this_extension)

        pan_3 = pan_2
        tilt_3 = tilt_2
        zoom_3 = zoom_2
        if pan_1 < pan_2:
            pan_3 = pan_2 + 30
        elif pan_1 > pan_2:
            pan_3 = pan_2 - 30
        if tilt_1 < tilt_2:
            tilt_3 = tilt_2 + 15
        elif tilt_1 > tilt_2:
            tilt_3 = tilt_2 - 15

        if pan_3 > 330:
            pan_3 = -330
        if pan_3 < -330:
            pan_3 = 330

        if tilt_3 < -30:
            tilt_3 = 30
        if tilt_3 > 30:
            tilt_3 = -30

        new_orientation = "{}-{}-{}".format(pan_3, tilt_3, zoom_3)
        
        return new_orientation if new_orientation in set(orientations) else neighbor_used_for_this_extension

    def get_coordinates_of_orientation_from_boxes(all_boxes):
        min_x, min_y, max_x, max_y = 0,0,0,0
        for box in all_boxes:
            (x1, y1, x2, y2) = box
            if x1 < min_x:
                min_x = x1
            if y1 < min_y:
                min_y = y1
            if x2 > max_x:
                max_x = x2
            if y2 > max_y:
                max_y = y2
        return min_x, min_y, max_x, max_y

    def get_center_of_list_of_boxes(all_boxes):
        min_x, min_y, max_x, max_y = get_coordinates_of_orientation_from_boxes(all_boxes)
        return (max_x - min_x) / 2, (max_y - min_y) / 2 


    def get_centroid_of_list_of_boxes(all_boxes):
        centroids = []
        for box in all_boxes:
            (x1, y1, x2, y2) = box
            centroid_x = (x1 + x2) / 2
            centroid_y = (y1 + y2) / 2
            centroids.append((centroid_x, centroid_y))
        centroid_x = sum(x for x, y in centroids) / len(centroids)
        centroid_y = sum(y for x, y in centroids) / len(centroids)
        return centroid_x, centroid_y

    def get_coordinates_of_line_corresponding_to_orientation_border(o, n, coordinates_of_o):
        min_x, min_y, max_x, max_y = coordinates_of_o
        # o is an orientation
        # n is another orientation that is adjacent
        # coordinates of o gives us the min_x,min_y max_x, max_y of o
        
        # find out which side of o n is on
        if extract_pan(o) == extract_pan(n):
            # n is on top or bottom of o
            current_tilt = extract_tilt(o)
            target_tilt = extract_tilt(n)
            if target_tilt > current_tilt:
                # n is above
                return min_x, min_y, max_x, min_y
            else:
                # n is below
                return min_x, max_y, max_x, max_y
        elif extract_tilt(o) == extract_tilt(n):
            # n is to the left or right of o
            current_pan = extract_pan(o)
            target_pan = extract_pan(n)
            if current_pan > target_pan:
                if current_pan - target_pan <= 180:
                    # Rotating left
                    return min_x, min_y, min_x, max_y
                # Rotating right
                return max_x, min_y, max_x, max_y
            else:
                if target_pan - current_pan <= 180:
                    # Rotating right
                    return max_x, min_y, max_x, max_y
                # Rotating left
                return min_x, min_y, min_x, max_y
        else:
            # n is diagonal
            current_tilt = extract_tilt(o)
            target_tilt = extract_tilt(n)
            current_pan = extract_pan(o)
            target_pan = extract_pan(n)
            if current_tilt < target_tilt:
                if current_pan < target_pan:
                    # n is at left bottom. tilt increases from bottom to top
                    return min_x, max_y, min_x, max_y
                else:
                    # n is at right bottom
                    return max_x, max_y, max_x, max_y
            else:
                if current_pan < target_pan:
                    # n is at top left
                    return min_x, min_y, min_x, min_y
                else:
                    # n is at top right
                    return max_x, min_y, max_x, min_y

        

    def fraction_of_box_areas_towards_neighbor(o, n, orientation_to_current_car_boxes, orientation_to_current_person_boxes):
        # we have an orientation and a neighbor
        # we know all the boxes within the orientation
        # if the boxes in o are in aggregate gathered nearer to n, n should get a boost
        # this function returns this metric. it returns 0 if the boxes are exactly 
        # in the center of o. a negative number of the boxes are away from n (more negative the farther away)
        # and a positive number if the boxes are closer to n (more positive if closer to n).

        # get centroid C of all boxes
        current_all_boxes = []
        if o in orientation_to_current_person_boxes:  
            current_all_boxes.extend(orientation_to_current_person_boxes[o])
        elif o in orientation_to_current_car_boxes:
            current_all_boxes.extend(orientation_to_current_car_boxes[o])
        current_all_boxes = [tuple(list(x)) for x in current_all_boxes]
        if len(current_all_boxes) == 0:
            return 0
        centroid_x, centroid_y = get_centroid_of_list_of_boxes(current_all_boxes)
        
        # get cumulative area of all boxes
        cumulative_area_of_all_boxes = evaluation_tools.get_nonoverlapping_sum_of_areas_of_list_of_bounding_boxes(current_all_boxes)

        # get center of o called c
        # center_x, center_y = get_center_of_list_of_boxes(current_all_boxes)
        center_x = 720.0
        center_y = 360.0
        
        neighbor_x1, neighbor_y1, neighbor_x2, neighbor_y2 = get_coordinates_of_line_corresponding_to_orientation_border(o, n, get_coordinates_of_orientation_from_boxes(current_all_boxes))
        # get distance of C to n
        distance_of_centroid_to_neighbor = distance_from_point_to_line(neighbor_x1, neighbor_y1, neighbor_x2, neighbor_y2, centroid_x, centroid_y)
        
        # get distance of c to n
        distance_of_center_to_neighbor = distance_from_point_to_line(neighbor_x1, neighbor_y1, neighbor_x2, neighbor_y2, center_x, center_y)

        # in addition to distance, we also account for avg area of bounding box. we take average here otherwise one large object can dominate (i.e. we would not be fair towards count queries. considering area is for detect, and considering num bounding boxes is for count)
        # return ((distance_of_center_to_neighbor-distance_of_centroid_to_neighbor)/distance_of_center_to_neighbor) * (cumulative_area_of_all_boxes / len(current_all_boxes))
        
        # option 2: didn't work if i added cumulative area
        if distance_of_center_to_neighbor == 0:
            return 0
        return ((distance_of_center_to_neighbor-distance_of_centroid_to_neighbor)/distance_of_center_to_neighbor)


    def is_formation_of_orientations_feasible(potential_set_of_shapes, all_orientations):
        # TODO: later we'll also want to determine ideal location to start and end

        # TODO: we now assume 500 degrees per second we can visit all the shapes for sure
        # contiguous if each orientation has at least one other neighbor within the shape
        # set_of_nodes_to_explore = copy.deepcopy(potential_set_of_shapes)
        if len(potential_set_of_shapes) == 0:
            return True 
        contiguous=False
        for starting_node in potential_set_of_shapes:
            set_of_nodes_to_visit_to_reach_everyone = copy.deepcopy(potential_set_of_shapes)
            set_of_nodes_to_visit = []
            set_of_nodes_to_visit.append(starting_node)
            while len(set_of_nodes_to_visit) > 0:
                if len(set_of_nodes_to_visit_to_reach_everyone) == 0:
                    # print(f"returning True feasibility for {potential_set_of_shapes}")
                    contiguous = True
                    break
                current_position = set_of_nodes_to_visit[0]
                set_of_nodes_to_visit.pop(0)
                if current_position in set_of_nodes_to_visit_to_reach_everyone:
                    set_of_nodes_to_visit_to_reach_everyone.remove(current_position)
                    neighbors = get_neighbors_outside_shape(current_position, [], all_orientations)    
                    for n in neighbors:
                        set_of_nodes_to_visit.append(n)
            if len(set_of_nodes_to_visit_to_reach_everyone) == 0:
                # print(f"returning True feasibility for {potential_set_of_shapes}")
                contiguous = True
                break

        MST_less_than_500 = len(potential_set_of_shapes) <= num_frames_to_keep

#        MST_gt_lb = len(potential_set_of_shapes) >= LOWER_ORIENTATION_BOUND
 #       if contiguous and MST_less_than_500:# and MST_gt_lb:
  #          from dfs_helper import find_least_cost_path
   #         path, cost = find_least_cost_path(potential_set_of_shapes)
    #        MST_less_than_500 = cost <= 200
            

            
        # print(f"returning False feasibility for {potential_set_of_shapes}, orientations: {all_orientations}")
        return contiguous and MST_less_than_500

    
    original_formation = current_formation

    # print(f"*****INPUT shape is {current_formation} and its feasibility is {is_formation_of_orientations_feasible(current_formation, orientations)}")
    scores_and_deltas_used = {}
    
    # for each peek orientation, decide whether to keep it or remove it
    avg_mike_factor = statistics.mean(list(orientation_to_current_mike_factor.values()))
    # print(f"{step_number}: clearing peek")
    for o in peek_orientations:
        if (o in orientation_to_current_mike_factor and orientation_to_current_mike_factor[o] < avg_mike_factor/4) or (o not in orientation_to_current_mike_factor): 
            if len(current_formation) >= num_frames_to_keep + 1:
                current_formation_as_set = set(current_formation)
                current_formation_as_set.discard(o)
                current_formation = list(current_formation_as_set)
    peek_orientations.clear()

    # use the current counts to indicate how much we'd benefit by staying in 
    # the same formation
    reward_from_current_orientations = copy.deepcopy(orientation_to_current_mike_factor)
    current_formation = sorted(current_formation, key=lambda x: reward_from_current_orientations[x], reverse=True)


    # Remove extra items from formation if current_formation is not feasible
    while not is_formation_of_orientations_feasible(current_formation, orientations):
        if len(current_formation) <= num_frames_to_keep:
            break
        if len(current_formation) <= num_frames_to_keep+ 1:
            break
        current_formation.pop(-1)

    while len(current_formation) > num_frames_to_keep and len(current_formation) >= num_frames_to_keep + 1:
        current_formation.pop(-1)

    current_best_scoring_o = current_formation[0]
    orientation_to_count_factor = {}
    for o in reward_from_current_orientations:
        if o not in orientation_to_historical_counts or len(orientation_to_historical_counts[o]) < 4:
            orientation_to_count_factor[o] = 0.3 * orientation_to_current_mike_factor[o]
        else:
            # did we visit it 4 times within the last 30 frames?
            if step_number - orientation_to_visited_step_numbers[o][-4] < 30:
                c1 = orientation_to_historical_counts[o][-1]
                c2 = orientation_to_historical_counts[o][-2]
                c3 = orientation_to_historical_counts[o][-3]
                c4 = orientation_to_historical_counts[o][-4]
                orientation_to_count_factor[o] = (c1-c2) * 0.7 + (c2-c3) * 0.2 + (c3-c4) * 0.1
            else:
                # dont use the historical count as it is stale
                orientation_to_count_factor[o] = 0.3 * orientation_to_current_mike_factor[o]
    orientation_to_potential = {}
    for o, v in reward_from_current_orientations.items():
        orientation_to_potential[o] = (0.3 * orientation_to_current_mike_factor[o]) + (0.7 * orientation_to_count_factor[o]) 
        # print(f"step:{step_number}. o:{o}, (0.3 * orientation_to_current_mike_factor[o]):{(0.3 * orientation_to_current_mike_factor[o])}, (0.7 * orientation_to_count_factor[o]):{(0.7 * orientation_to_count_factor[o]) }")
    
    # print(f"sorting current formation {current_formation} by mike factor alone")
    # current_formation = sorted(current_formation, key=lambda x: orientation_to_current_mike_factor[x])

    
    current_formation = sorted(current_formation, key=lambda x: orientation_to_potential[x])
    # print(f"step:{step_number} sorting current formation {current_formation} by potential instead of mike factor alone")
    

    orientation_to_candidate_neighbors = {}
    for o in orientation_to_potential:
        orientation_to_candidate_neighbors[o] = get_neighbors_outside_shape(o, current_formation, orientations)    

    neighbor_to_viability_score = {}
    neighbor_to_number_of_touching_boundary_objects = {}
    for o in orientation_to_potential:
        neighbors = orientation_to_candidate_neighbors[o]
        for n in neighbors:
            if n not in neighbor_to_viability_score:
                neighbor_to_viability_score[n] = 0
            if n not in neighbor_to_number_of_touching_boundary_objects:
                neighbor_to_number_of_touching_boundary_objects[n] = 0
            neighbor_to_number_of_touching_boundary_objects[n] += 1
            neighbor_to_viability_score[n] += fraction_of_box_areas_towards_neighbor(o, n, orientation_to_current_car_boxes, orientation_to_current_person_boxes)
    # input(f"neighbor to viability score is {neighbor_to_viability_score}")
    for o in orientation_to_potential:
        orientation_to_candidate_neighbors[o] = sorted(orientation_to_candidate_neighbors[o], key=lambda x: neighbor_to_viability_score[x] * neighbor_to_number_of_touching_boundary_objects[x], reverse=True)
    


    scores_and_deltas_used = {
        "current shape scores": reward_from_current_orientations,
        "potential for border": orientation_to_potential,
        "orientation_to_current_mike_factor": orientation_to_current_mike_factor 
    }

    left_index = 0
    right_index = len(current_formation) - 1
    continue_modifying_shape = True
    num_prior_expansions_to_this_right = 0
    number_of_neighbors_used_in_this_right_index = 0
    number_of_changes = 0
    set_of_orientations_in_new_shape = set(current_formation)
    orientations_to_number_of_expansions = {}
    
    while left_index < right_index and continue_modifying_shape:
        if current_formation[left_index] == current_best_scoring_o:
            left_index += 1
        if only_added_orientation_last_frame(current_formation[left_index], orientation_to_visited_step_numbers, step_number):
            left_index += 1
        if left_index >= right_index:
            break
        # print(f"left index is {left_index}, right_index is {right_index}")
        # print(f"set_of_orientations_in_new_shape is {set_of_orientations_in_new_shape}")
        lower_score = reward_from_current_orientations[current_formation[left_index]]
        higher_score = reward_from_current_orientations[current_formation[right_index]]
        # print(f"lower_score={lower_score}, higher_score={higher_score}")
        if not can_swap_right(num_prior_expansions_to_this_right, lower_score, higher_score):
            # print(f"can_swap_right with {num_prior_expansions_to_this_right} is false")
            # print("1")
            if False and can_expand_right(num_prior_expansions_to_this_right, lower_score, higher_score):
                # not a big enough threshold to remove old one confidently and add 
                # but is there some gap between higher and lowest to warrant exploration 
                # near the higher?
                print('3')
                if len(orientation_to_candidate_neighbors[current_formation[right_index]]) > num_prior_expansions_to_this_right:
                    potential_new_shape = copy.deepcopy(set_of_orientations_in_new_shape)
                    is_new_shape_feasible = True
                    number_of_neighbors_used_in_this_right_index = 0
                    while orientation_to_candidate_neighbors[current_formation[right_index]][number_of_neighbors_used_in_this_right_index] in potential_new_shape:
                        number_of_neighbors_used_in_this_right_index += 1
                        if number_of_neighbors_used_in_this_right_index >= len(orientation_to_candidate_neighbors[current_formation[right_index]]):
                            break
                    if number_of_neighbors_used_in_this_right_index >= len(orientation_to_candidate_neighbors[current_formation[right_index]]):
                        is_new_shape_feasible = False
                        break
                    if number_of_neighbors_used_in_this_right_index < len(orientation_to_candidate_neighbors[current_formation[right_index]]):
                        potential_new_shape.add(orientation_to_candidate_neighbors[current_formation[right_index]][number_of_neighbors_used_in_this_right_index])
                    if (is_new_shape_feasible) and is_formation_of_orientations_feasible(potential_new_shape, orientations):
                        num_prior_expansions_to_this_right += 1
                        right_orientation = current_formation[right_index]
                        if right_orientation not in orientations_to_number_of_expansions:
                            orientations_to_number_of_expansions[right_orientation] = 0
                        orientations_to_number_of_expansions[right_orientation] += 1
                        number_of_changes += 1
                        set_of_orientations_in_new_shape = copy.deepcopy(potential_new_shape)
                    else:
                        # couldn't swap right
                        # couldn't expand right
                        right_index -= 1
                        number_of_neighbors_used_in_this_right_index = 0
                        num_prior_expansions_to_this_right = 0
                
            else:
                # couldn't swap right
                # couldn't expand right
                right_index -= 1
                number_of_neighbors_used_in_this_right_index = 0
                num_prior_expansions_to_this_right = 0
        else:
            # print(f"can_swap_right with {num_prior_expansions_to_this_right} is true")
            if len(orientation_to_candidate_neighbors[current_formation[right_index]]) > num_prior_expansions_to_this_right:
                # print(f"when trying to expand: {set_of_orientations_in_new_shape}")
                potential_new_shape = copy.deepcopy(set_of_orientations_in_new_shape)
                is_new_shape_feasible = True
                # print(f"removing {current_formation[left_index]}")
                potential_new_shape.remove(current_formation[left_index])
                number_of_neighbors_used_in_this_right_index = 0
                while orientation_to_candidate_neighbors[current_formation[right_index]][number_of_neighbors_used_in_this_right_index] in potential_new_shape:
                    # print(f"adding {orientation_to_candidate_neighbors[current_formation[right_index]][number_of_neighbors_used_in_this_right_index]}")
                
                    # print(f"but {orientation_to_candidate_neighbors[current_formation[right_index]][number_of_neighbors_used_in_this_right_index]} is in {potential_new_shape}")
                    number_of_neighbors_used_in_this_right_index += 1
                    if number_of_neighbors_used_in_this_right_index >= len(orientation_to_candidate_neighbors[current_formation[right_index]]):
                        is_new_shape_feasible = False
                        break
                    # print(f"going to try if {orientation_to_candidate_neighbors[current_formation[right_index]][number_of_neighbors_used_in_this_right_index]} is in {potential_new_shape}")
                neighbor_used_for_this_extension = None
                if number_of_neighbors_used_in_this_right_index < len(orientation_to_candidate_neighbors[current_formation[right_index]]):
                    neighbor_used_for_this_extension = orientation_to_candidate_neighbors[current_formation[right_index]][number_of_neighbors_used_in_this_right_index]
                
                    potential_new_shape.add(neighbor_used_for_this_extension)
                # print(f"when evaluating {potential_new_shape}, is_new_shape_feasible is {is_new_shape_feasible}")
                if (is_new_shape_feasible) and is_formation_of_orientations_feasible(potential_new_shape, orientations):
                    # print(f"step:{step_number} removing {current_formation[left_index]} adding neighbor of {current_formation[right_index]}")
                    # print(f"{lower_score}, {higher_score}")
                    num_prior_expansions_to_this_right += 1
                    left_index += 1
                    right_orientation = current_formation[right_index]
                    if right_orientation not in orientations_to_number_of_expansions:
                        orientations_to_number_of_expansions[right_orientation] = 0
                    orientations_to_number_of_expansions[right_orientation] += 1
                    number_of_changes += 1
                    set_of_orientations_in_new_shape = copy.deepcopy(potential_new_shape)
                    if neighbor_used_for_this_extension is not None and neighbor_to_number_of_touching_boundary_objects[neighbor_used_for_this_extension] >= 2:
                        # do a second hop
                        # print(f"********")
                        # print(f"********")
                        # print(f"neighbor_to_viability_score[neighbor_used_for_this_extension]:{neighbor_to_viability_score[neighbor_used_for_this_extension]}")
                        second_hop_neighbor = extrapolate_orientation(right_orientation, neighbor_used_for_this_extension, orientations)
                        # print(f"r:{right_orientation}, n:{neighbor_used_for_this_extension}, s:{second_hop_neighbor}")
                        potential_new_shape.add(second_hop_neighbor)
                        if is_formation_of_orientations_feasible(potential_new_shape, orientations):
                            set_of_orientations_in_new_shape = copy.deepcopy(potential_new_shape)
                            # print(f"{step_number}: added peek {second_hop_neighbor}")
                            peek_orientations.add(second_hop_neighbor)
                    # print(f"it is feasible so updated shape to be {set_of_orientations_in_new_shape}")
                else:
                    # print(f"it is infeasible so set_of_orientations_in_new_shape remains {set_of_orientations_in_new_shape}")
                    right_index -= 1
                    num_prior_expansions_to_this_right = 0
                    number_of_neighbors_used_in_this_right_index = 0
                    # print(f"shape is infeasible")
            else: 
                # print(f"right has no more neighbors")
                right_index -= 1
                num_prior_expansions_to_this_right = 0
                number_of_neighbors_used_in_this_right_index = 0
        if (not is_formation_of_orientations_feasible(set_of_orientations_in_new_shape, orientations)):
            # print(f"stop modifying shape due to too many changes or infeasible shape")
            continue_modifying_shape = False
        # add a dampener on too many hasty changes
        if step_number <= 3 and len(set(set_of_orientations_in_new_shape) - set(current_formation)) >= 1:
            continue_modifying_shape = False


    current_formation = list(set_of_orientations_in_new_shape)
    # print(f"current formation is {current_formation}")
    # current_formation = reset_zoom_factors(current_formation, anchor_orientation)
    current_formation, zoom_explorations_in_progress = add_zoom_factors(current_formation, orientation_to_current_car_boxes, orientation_to_current_person_boxes, zoom_explorations_in_progress)
    # input(f"after adding zoom factors {current_formation}")
    for o in current_formation:
        if o not in orientation_to_visited_step_numbers:
            orientation_to_visited_step_numbers[o] = []
        orientation_to_visited_step_numbers[o].append(step_number)
    

#    plus_formation = []
#    plus_formation.append(anchor_orientation)
#    plus_formation.append(rotate_down(anchor_orientation, orientations))
#    plus_formation.append(rotate_left(anchor_orientation, orientations))
#    plus_formation.append(rotate_right(anchor_orientation, orientations))
#    plus_formation.append(rotate_up(anchor_orientation, orientations))
#    for p in plus_formation:
#        if len(current_formation) < num_frames_to_keep and  p not in current_formation:
#            current_formation.append(p)

    formation_to_use = current_formation                
    return original_formation, current_formation, formation_to_use, scores_and_deltas_used, step_number + 1, zoom_explorations_in_progress

