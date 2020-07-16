import pickle
import json
import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET

def _darknet_box_to_coco_bbox(box, img_width, img_height):
  # darknet  box format: normalized (x_ctr, y_ctr, w, h)
  # coco box format: unnormalized (xmin, ymin, width, height)

  box_width = round(box[2] * img_width, 2)
  box_height = round(box[3] * img_height, 2)
  box_x_ctr = box[0] * img_width
  box_y_ctr = box[1] * img_height

  xmin = round(box_x_ctr - box_width / 2., 2)
  xmax = round(box_x_ctr + box_width / 2., 2)
  ymin = round(box_y_ctr - box_height / 2., 2)
  ymax = round(box_y_ctr + box_height / 2., 2)

  bbox = np.array([xmin, ymin, box_width, box_height], dtype=np.float32)
  print (bbox)
  return [xmin, ymin, box_width, box_height]

def _pascal_box_to_coco_bbox(box):
  # pascal box format: unnormalized (xmin, ymin, xmax, ymax)
  # coco   box format: unnormalized (xmin, ymin, width, height)
  width = box[2] - box[0] + 1
  height = box[3] - box[1] + 1
  return [box[0], box[1], width, height]  

DATA_PATH = '/media/yehao/data/COCO+VOC_person_filter/'

SPLITS = ['train', 'val']
#SPLITS = ['train']

cats = ['person']

cat_ids = {cat: i for i, cat in enumerate(cats)}

cat_info = []
for i, cat in enumerate(cats):
  cat_info.append({'name': cat, 'id': i})

for SPLIT in SPLITS:
    image_set_path = DATA_PATH + '{}.txt'.format(SPLIT)
    print (image_set_path)
    anno_dir = DATA_PATH + 'Annotations/'

    ret = {'images': [], 'annotations': [], "categories": cat_info}

    i = 0
    for line in open(image_set_path, 'r'):
        line = line.strip()
        img_name = line + '.jpg'
        xml_name = line + '.xml'
        img_name = DATA_PATH + 'imgs/' + img_name

        i += 1
        #print (i)
        image_id = int(i)
        #print (image_id)
        image_info = {'file_name': '{}'.format(img_name), 'id': image_id}
        #print (image_info)
        #print (img_name)
        # coco annotation format
        ret['images'].append(image_info)

        # anno path
        anno_path = anno_dir + xml_name
        tree = ET.ElementTree(file = anno_path)
        root = tree.getroot()

        for elem in root.findall('object'):
                child = elem.findall('bndbox')

                # get class
                category = elem[0].text
                # only get person instance
                if category == 'person':
                      cat_id = cat_ids[category]

                      for j in range(0, 4):		                            
                          if child[0][j].tag == 'xmin':
                              #xmin = int(child[0][j].text)
                              xmin = float(child[0][j].text)
                          if child[0][j].tag == 'ymin':
                              ymin = float(child[0][j].text)
                          if child[0][j].tag == 'xmax':
                              xmax = float(child[0][j].text)
                          if child[0][j].tag == 'ymax':
                              ymax = float(child[0][j].text) 

                      if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
                          print (category, xmin, ymin, xmax, ymax) 
                          print (img_name)


                      box = [xmin, ymin, xmax, ymax] 
                      [xmin, ymin, box_width, box_height] = _pascal_box_to_coco_bbox(box) 

                      truncated = 0 
                      occluded = 0 
                      is_crowd = 0

                      ann = {'image_id': image_id,
                                  'id': int(len(ret['annotations']) + 1),
                                  'category_id': int(cat_id),
                                  'bbox': _pascal_box_to_coco_bbox(box),
                                  'truncated': truncated,
                                  'occluded': occluded,
                                  'iscrowd': is_crowd,
                                  'area': box_width * box_height}
                      ret['annotations'].append(ann)
		            

    print ('sum: ', i)

    out_path = '{}/annotations/{}.json'.format(DATA_PATH, SPLIT)
    json.dump(ret, open(out_path, 'w'))

    
