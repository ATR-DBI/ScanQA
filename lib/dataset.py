""" 
Modified from: https://github.com/daveredrum/ScanRefer/blob/master/lib/dataset.py
"""

import re
import os
import sys
import time
import h5py
import json
import pickle
import numpy as np
import multiprocessing as mp

#from sklearn import preprocessing
from torch.utils.data import Dataset
from data.scannet.model_util_scannet import ScannetDatasetConfig

sys.path.append(os.path.join(os.getcwd(), 'lib')) # HACK add the lib folder
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from data.scannet.model_util_scannet import ScannetDatasetConfig, rotate_aligned_boxes_along_axis

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, 'scannetv2-labels.combined.tsv')
MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, 'glove.p')


def get_answer_score(freq):
    if freq == 0:
        return .0
    elif freq == 1:
        return .3
    elif freq == 2:
        return .6
    elif freq == 3:
        return .9
    else:
        return 1.

class ScannetQADatasetConfig(ScannetDatasetConfig):
    def __init__(self):
        super().__init__()
        self.num_answers = -1

class Answer(object):
    def __init__(self, answers=None, unk_token='<unk>', ignore_idx=-100):
        if answers is None:
            answers = []
        self.unk_token = unk_token
        self.ignore_idx = ignore_idx
        self.vocab = {x: i for i, x in enumerate(answers)}
        self.rev_vocab = dict((v, k) for k, v in self.vocab.items())

    def itos(self, i):
        if i == self.ignore_idx:
            return self.unk_token
        return self.rev_vocab[i]

    def stoi(self, v):
        if v not in self.vocab:
            #return self.vocab[self.unk_token]
            return self.ignore_idx
        return self.vocab[v]

    def __len__(self):
        return len(self.vocab)    


class ScannetQADataset(Dataset):
    def __init__(self, scanqa, scanqa_all_scene, 
            use_unanswerable=False,
            answer_cands=None,
            answer_counter=None,
            answer_cls_loss='ce',
            split='train', 
            num_points=40000,
            use_height=False, 
            use_color=False, 
            use_normal=False, 
            use_multiview=False, 
            tokenizer=None,
            augment=False,
            debug=False,
        ):

        self.debug = debug
        self.all_data_size = -1
        self.answerable_data_size = -1

        self.answer_features = None
        self.use_unanswerable = use_unanswerable

        if split == 'train':
            # remove unanswerble qa samples for training
            self.all_data_size = len(scanqa)
            if use_unanswerable: 
                self.scanqa = scanqa
            else:
                self.scanqa = [data for data in scanqa if len(set(data['answers']) & set(answer_cands)) > 0]
            self.answerable_data_size = len(self.scanqa)
            print('all train:', self.all_data_size)
            print('answerable train', self.answerable_data_size)
        elif split == 'val':
            self.all_data_size = len(scanqa)
            if use_unanswerable:
                self.scanqa = scanqa
            else:
                self.scanqa = [data for data in scanqa if len(set(data['answers']) & set(answer_cands)) > 0]
                
            self.answerable_data_size = len(self.scanqa)
            print('all val:', self.all_data_size)
            print('answerable val', self.answerable_data_size)
        elif split == 'test':
            self.scanqa = scanqa

        self.scanqa_all_scene = scanqa_all_scene # all scene_ids in scanqa
        self.answer_cls_loss = answer_cls_loss
        self.answer_cands = answer_cands
        self.answer_counter = answer_counter
        self.answer_vocab = Answer(answer_cands)
        self.num_answers = 0 if answer_cands is None else len(answer_cands) 

        self.split = split
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment

        # tokenize a question to tokens
        scene_ids = sorted(set(record['scene_id'] for record in self.scanqa))
        self.scene_id_to_number = {scene_id:int(''.join(re.sub('scene', '', scene_id).split('_'))) for scene_id in scene_ids}
        self.scene_number_to_id = {v: k for k, v in self.scene_id_to_number.items()}

        self.use_bert_embeds = False
        if tokenizer is None:
            from spacy.tokenizer import Tokenizer
            from spacy.lang.en import English
            nlp = English()
            # Create a blank Tokenizer with just the English vocab
            spacy_tokenizer = Tokenizer(nlp.vocab)
            
            def tokenize(sent):
                sent = sent.replace('?', ' ?')
                return [token.text for token in spacy_tokenizer(sent)]

            for record in self.scanqa:
                record.update(token=tokenize(record['question'])) 
        else:
            self.use_bert_embeds = True
            for record in self.scanqa:
                record.update(token=tokenizer(record['question'], return_tensors='np'))
            
        # load data
        self._load_data()
        self.multiview_data = {}


    def __len__(self):
        return len(self.scanqa)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scanqa[idx]['scene_id']
        if self.split != 'test':
            object_ids = self.scanqa[idx]['object_ids']
            object_names = [' '.join(object_name.split('_')) for object_name in self.scanqa[idx]['object_names']]
        else:            
            object_ids = None
            object_names = None            

        question_id = self.scanqa[idx]['question_id']
        answers = self.scanqa[idx]['answers']

        answer_cats = np.zeros(self.num_answers) 
        answer_inds = [self.answer_vocab.stoi(answer) for answer in answers]

        if self.answer_counter is not None:        
            answer_cat_scores = np.zeros(self.num_answers)
            for answer, answer_ind in zip(answers, answer_inds):
                if answer_ind < 0:
                    continue                    
                answer_cats[answer_ind] = 1
                answer_cat_score = get_answer_score(self.answer_counter.get(answer, 0))
                answer_cat_scores[answer_ind] = answer_cat_score

            if not self.use_unanswerable:
                assert answer_cats.sum() > 0
                assert answer_cat_scores.sum() > 0
        else:
            raise NotImplementedError

        answer_cat = answer_cats.argmax()

        #
        # get language features
        #
        if self.use_bert_embeds:
            lang_feat = self.lang[scene_id][question_id]
            lang_feat['input_ids'] = lang_feat['input_ids'].astype(np.int64)
            lang_feat['attention_mask'] = lang_feat['attention_mask'].astype(np.float32)
            if 'token_type_ids' in lang_feat:
                lang_feat['token_type_ids'] = lang_feat['token_type_ids'].astype(np.int64)
            lang_len = self.scanqa[idx]['token']['input_ids'].shape[1]
        else:
            lang_feat = self.lang[scene_id][question_id]
            lang_len = len(self.scanqa[idx]['token'])

        lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_TEXT_LEN else CONF.TRAIN.MAX_TEXT_LEN
        #
        # get point cloud features
        #
        mesh_vertices = self.scene_data[scene_id]['mesh_vertices']
        instance_labels = self.scene_data[scene_id]['instance_labels']
        semantic_labels = self.scene_data[scene_id]['semantic_labels']
        instance_bboxes = self.scene_data[scene_id]['instance_bboxes']

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3]
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1) # p (50000, 7)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1)

        '''
        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA + '.hdf5', 'r', libver='latest')
            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)
        '''

        #'''
        if self.use_multiview:
            # load multiview database
            enet_feats_file = os.path.join(MULTIVIEW_DATA, scene_id) + '.pkl'
            multiview = pickle.load(open(enet_feats_file, 'rb'))
            point_cloud = np.concatenate([point_cloud, multiview],1) # p (50000, 135)
        #'''

        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]
        
        # ------------------------------- LABELS ------------------------------    
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        ref_box_label = np.zeros(MAX_NUM_OBJ) # bbox label for reference target

        ref_center_label = np.zeros(3) # bbox center for reference target
        ref_heading_class_label = 0
        ref_heading_residual_label = 0
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3) # bbox size residual for reference target

        if self.split != 'test':
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox,:] = instance_bboxes[:MAX_NUM_OBJ,0:6]

            point_votes = np.zeros([self.num_points, 3])
            point_votes_mask = np.zeros(self.num_points)

            # ------------------------------- DATA AUGMENTATION ------------------------------        
            if self.augment and not self.debug:
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    point_cloud[:,0] = -1 * point_cloud[:,0]
                    target_bboxes[:,0] = -1 * target_bboxes[:,0]                
                    
                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    point_cloud[:,1] = -1 * point_cloud[:,1]
                    target_bboxes[:,1] = -1 * target_bboxes[:,1]                                

                # Rotation along X-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotx(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, 'x')

                # Rotation along Y-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = roty(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, 'y')

                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, 'z')

                # Translation
                point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)

            # compute votes *AFTER* augmentation
            # generate votes
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered 
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label. 
            for i_instance in np.unique(instance_labels):            
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                # find the semantic label            
                if semantic_labels[ind[0]] in DC.nyu40ids:
                    x = point_cloud[ind,:3]
                    center = 0.5*(x.min(0) + x.max(0))
                    point_votes[ind, :] = center - x
                    point_votes_mask[ind] = 1.0
            point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
            
            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]

            # construct the reference target label for each bbox
            ref_box_label = np.zeros(MAX_NUM_OBJ)

            for i, gt_id in enumerate(instance_bboxes[:num_bbox,-1]): 
                if gt_id == object_ids[0]:
                    ref_box_label[i] = 1
                    ref_center_label = target_bboxes[i, 0:3]
                    ref_heading_class_label = angle_classes[i]
                    ref_heading_residual_label = angle_residuals[i]
                    ref_size_class_label = size_classes[i]
                    ref_size_residual_label = size_residuals[i]

            
            assert ref_box_label.sum() > 0
        else:
            num_bbox = 1
            point_votes = np.zeros([self.num_points, 9]) # make 3 votes identical 
            point_votes_mask = np.zeros(self.num_points)

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
        except KeyError:
            pass

        object_name = None if object_names is None else object_names[0]
        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17

        data_dict = {}
        if self.use_bert_embeds:
            data_dict['lang_feat'] = lang_feat
        else:
            data_dict['lang_feat'] = lang_feat.astype(np.float32) # language feature vectors
        data_dict['point_clouds'] = point_cloud.astype(np.float32) # point cloud data including features
        data_dict['lang_len'] = np.array(lang_len).astype(np.int64) # length of each description
        data_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3] # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict['heading_class_label'] = angle_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict['heading_residual_label'] = angle_residuals.astype(np.float32) # (MAX_NUM_OBJ,)
        data_dict['size_class_label'] = size_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict['size_residual_label'] = size_residuals.astype(np.float32) # (MAX_NUM_OBJ, 3)
        data_dict['num_bbox'] = np.array(num_bbox).astype(np.int64)
        data_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64) # (MAX_NUM_OBJ,) semantic class index
        data_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32) # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict['vote_label'] = point_votes.astype(np.float32) # 
        data_dict['vote_label_mask'] = point_votes_mask.astype(np.int64) # point_obj_mask (gf3d)
        data_dict['scan_idx'] = np.array(idx).astype(np.int64)
        data_dict['pcl_color'] = pcl_color
        data_dict['ref_box_label'] = ref_box_label.astype(np.int64) # (MAX_NUM_OBJ,) # 0/1 reference labels for each object bbox

        data_dict['ref_center_label'] = ref_center_label.astype(np.float32) # (3,)
        data_dict['ref_heading_class_label'] = np.array(int(ref_heading_class_label)).astype(np.int64) # (MAX_NUM_OBJ,)
        data_dict['ref_heading_residual_label'] = np.array(int(ref_heading_residual_label)).astype(np.int64) # (MAX_NUM_OBJ,)
        data_dict['ref_size_class_label'] = np.array(int(ref_size_class_label)).astype(np.int64) # (MAX_NUM_OBJ,)
        data_dict['ref_size_residual_label'] = ref_size_residual_label.astype(np.float32) 
        data_dict['object_cat'] = np.array(object_cat).astype(np.int64)

        data_dict['scene_id'] = np.array(int(self.scene_id_to_number[scene_id])).astype(np.int64)
        if type(question_id) == str:
            data_dict['question_id'] = np.array(int(question_id.split('-')[-1])).astype(np.int64)
        else:
            data_dict['question_id'] = np.array(int(question_id)).astype(np.int64)
        data_dict['pcl_color'] = pcl_color
        data_dict['load_time'] = time.time() - start
        data_dict['answer_cat'] = np.array(int(answer_cat)).astype(np.int64) # 1
        data_dict['answer_cats'] = answer_cats.astype(np.int64) # num_answers
        if self.answer_cls_loss == 'bce' and self.answer_counter is not None:
            data_dict['answer_cat_scores'] = answer_cat_scores.astype(np.float32) # num_answers
        return data_dict

    
    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanqa:
            scene_id = data['scene_id']

            for object_id, object_name in zip(data['object_ids'], data['object_names']):
                object_id = data['object_ids'][0]
                object_name = ' '.join(object_name.split('_'))

                if scene_id not in all_sem_labels:
                    all_sem_labels[scene_id] = []

                if scene_id not in cache:
                    cache[scene_id] = {}

                if object_id not in cache[scene_id]:
                    cache[scene_id][object_id] = {}
                    try:
                        all_sem_labels[scene_id].append(self.raw2label[object_name])
                    except KeyError:
                        all_sem_labels[scene_id].append(17)

        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanqa:
            scene_id = data['scene_id']
            question_id = data['question_id']

            unique_multiples = []
            for object_id, object_name in zip(data['object_ids'], data['object_names']):
                object_id = data['object_ids'][0]
                object_name = ' '.join(object_name.split('_'))
                try:
                    sem_label = self.raw2label[object_name]
                except KeyError:
                    sem_label = 17

                unique_multiple_ = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1
                unique_multiples.append(unique_multiple_)

            unique_multiple = max(unique_multiples)

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            unique_multiple_lookup[scene_id][question_id] = unique_multiple

        return unique_multiple_lookup

    def _tranform_text_glove(self, token_type='token'):
        with open(GLOVE_PICKLE, 'rb') as f:
            glove = pickle.load(f)

        lang = {}
        for data in self.scanqa:
            scene_id = data['scene_id']
            question_id = data['question_id']

            if scene_id not in lang:
                lang[scene_id] = {}

            if question_id in lang[scene_id]:
                continue

            # tokenize the description
            tokens = data[token_type]
            embeddings = np.zeros((CONF.TRAIN.MAX_TEXT_LEN, 300))
            # tokens = ['sos'] + tokens + ['eos']
            # embeddings = np.zeros((CONF.TRAIN.MAX_TEXT_LEN + 2, 300))
            for token_id in range(CONF.TRAIN.MAX_TEXT_LEN):
                if token_id < len(tokens):
                    token = tokens[token_id]
                    if token in glove:
                        embeddings[token_id] = glove[token]
                    else:
                        embeddings[token_id] = glove['unk']

            # store
            lang[scene_id][question_id] = embeddings

        return lang

    def _tranform_text_bert(self, token_type='token'):
        lang = {}

        def pad_tokens(tokens):
            N = CONF.TRAIN.MAX_TEXT_LEN - 2 
            if tokens.ndim == 2:
                tokens = tokens[0]
            padded_tokens = np.zeros(CONF.TRAIN.MAX_TEXT_LEN)
            tokens = np.append(tokens[:-1][:N+1], tokens[-1:])
            padded_tokens[:len(tokens)] = tokens
            return padded_tokens

        for data in self.scanqa:
            scene_id = data['scene_id']
            question_id = data['question_id']

            if scene_id not in lang:
                lang[scene_id] = {}

            if question_id in lang[scene_id]:
                continue

            # for BERT
            if 'token_type_ids' in data[token_type]:
                padded_input_ids = pad_tokens(data[token_type]['input_ids'])
                padded_token_type_ids = pad_tokens(data[token_type]['token_type_ids'])
                padded_attention_mask = pad_tokens(data[token_type]['attention_mask'])
                # store
                lang[scene_id][question_id] = {
                    'input_ids': padded_input_ids, 
                    'token_type_ids': padded_token_type_ids,
                    'attention_mask': padded_attention_mask,
                }
            else: # for DistillBERT
                padded_input_ids = pad_tokens(data[token_type]['input_ids'])
                padded_attention_mask = pad_tokens(data[token_type]['attention_mask'])
                lang[scene_id][question_id] = {
                    'input_ids': padded_input_ids, 
                    'attention_mask': padded_attention_mask,
                }

        return lang


    def _load_data(self):
        print('loading data...')
        # load language features
        if self.use_bert_embeds:
            self.lang = self._tranform_text_bert('token')
        else:
            self.lang = self._tranform_text_glove('token')

        # add scannet data
        self.scene_list = sorted(list(set([data['scene_id'] for data in self.scanqa])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            self.scene_data[scene_id]['mesh_vertices'] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+'_aligned_vert.npy') # axis-aligned
            self.scene_data[scene_id]['instance_labels'] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+'_ins_label.npy')
            self.scene_data[scene_id]['semantic_labels'] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+'_sem_label.npy')
            self.scene_data[scene_id]['instance_bboxes'] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+'_aligned_bbox.npy')

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.label2raw = {v: k for k, v in self.raw2label.items()}
        if self.split != 'test':
            self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]
        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox
