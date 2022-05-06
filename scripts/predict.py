import os
import sys
import json
import pickle
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
from attrdict import AttrDict
from transformers import AutoTokenizer, AutoConfig

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

from lib.config import CONF
from lib.dataset import ScannetQADataset
from lib.ap_helper import parse_predictions
from lib.loss_helper import get_loss
from models.qa_module import ScanQA
from utils.box_util import get_3d_box
from utils.misc import overwrite_config
from data.scannet.model_util_scannet import ScannetDatasetConfig

project_name = "ScanQA_v1.0"


def get_dataloader(args, scanqa, all_scene_list, split, config):
    answer_vocab_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "answer_vocab.json")
    answer_counter = json.load(open(answer_vocab_path))
    answer_cands = sorted(answer_counter.keys())
    config.num_answers = len(answer_cands)    

    print("using {} answers".format(config.num_answers))

    if 'bert-' in args.tokenizer_name: 
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = None    

    dataset = ScannetQADataset(
        scanqa=scanqa, 
        scanqa_all_scene=all_scene_list, 
        use_unanswerable=True,         
        answer_cands=answer_cands,
        answer_counter=answer_counter,
        answer_cls_loss=args.answer_cls_loss,
        split=split, 
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color,         
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        tokenizer=tokenizer,
    )
    print("predict for {} samples".format(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataset, dataloader


def get_model(args, config):
    # load tokenizer model
    if "bert-" in args.tokenizer_name:
        bert_model_name = args.tokenizer_name
        bert_config = AutoConfig.from_pretrained(bert_model_name)
        if hasattr(bert_config, "hidden_size"):
            lang_emb_size = bert_config.hidden_size
        else:
            # for distllbert
            lang_emb_size = bert_config.dim
    else:
        bert_model_name = None
        lang_emb_size = 300 # glove emb_size

    # load model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)

    model = ScanQA(
        num_answers=config.num_answers,
        # proposal
        input_feature_dim=input_channels,            
        num_object_class=config.num_class, 
        num_heading_bin=config.num_heading_bin,
        num_size_cluster=config.num_size_cluster,
        mean_size_arr=config.mean_size_arr,
        num_proposal=args.num_proposals, 
        seed_feat_dim=args.seed_feat_dim,
        proposal_size=args.proposal_size,
        pointnet_width=args.pointnet_width,
        pointnet_depth=args.pointnet_depth,        
        vote_radius=args.vote_radius, 
        vote_nsample=args.vote_nsample,            
        # qa
        #answer_cls_loss="ce",
        answer_pdrop=args.answer_pdrop,
        mcan_num_layers=args.mcan_num_layers,
        mcan_num_heads=args.mcan_num_heads,
        mcan_pdrop=args.mcan_pdrop,
        mcan_flat_mlp_size=args.mcan_flat_mlp_size, 
        mcan_flat_glimpses=args.mcan_flat_glimpses,
        mcan_flat_out_size=args.mcan_flat_out_size,
        # lang
        lang_use_bidir=args.lang_use_bidir,
        lang_num_layers=args.lang_num_layers,
        lang_emb_size=lang_emb_size,
        lang_pdrop=args.lang_pdrop,
        bert_model_name=bert_model_name,
        freeze_bert=args.freeze_bert,
        finetune_bert_last_layer=args.finetune_bert_last_layer,
        # common
        hidden_size=args.hidden_size,
        # option
        use_object_mask=(not args.no_object_mask),
        use_lang_cls=(not args.no_lang_cls),
        use_reference=(not args.no_reference),
        use_answer=(not args.no_answer),            
        )

    model_name = "model.pth"
    model_path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
    print('loading model from:', model_path)
    # to CUDA
    model = model.cuda()
    model.load_state_dict(torch.load(model_path)) #, strict=False)
    model.eval()

    return model

def get_scanqa(args):
    scanqa = json.load(open(os.path.join(CONF.PATH.SCANQA, project_name + "_"+args.test_type+".json")))
    scene_list = sorted(list(set([data["scene_id"] for data in scanqa])))
    scanqa = [data for data in scanqa if data["scene_id"] in scene_list]
    return scanqa, scene_list

def predict(args):
    print("predict bounding boxes...")
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    print("preparing data...")
    scanqa, scene_list = get_scanqa(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanqa, scene_list, "test", DC)
    dataset = dataloader.dataset
    scanqa = dataset.scanqa

    # model
    model = get_model(args, DC)

    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    } if not args.no_nms else None

    if args.no_detection:
        POST_DICT = None

    # predict
    print("predicting...")
    pred_bboxes = []
    for data_dict in tqdm(dataloader):
        # move to cuda
        for key in data_dict:
            if type(data_dict[key]) is dict:
                data_dict[key] = {k:v.cuda() for k, v in data_dict[key].items()}
            else:
                data_dict[key] = data_dict[key].cuda()            

        # feed
        with torch.no_grad():        
            data_dict = model(data_dict)
            _, data_dict = get_loss(
                data_dict=data_dict, 
                config=DC, 
                detection=False,
                use_reference=not args.no_reference, 
                use_lang_classifier=not args.no_lang_cls,
                use_answer=(not args.no_answer),
            )

        objectness_preds_batch = torch.argmax(data_dict['objectness_scores'], 2).long()

        if POST_DICT:
            _ = parse_predictions(data_dict, POST_DICT)
            nms_masks = torch.LongTensor(data_dict['pred_mask']).cuda()
            # construct valid mask
            pred_masks = (nms_masks * objectness_preds_batch == 1).float()
        else:
            # construct valid mask
            pred_masks = (objectness_preds_batch == 1).float()

        # bbox prediction
        pred_ref = torch.argmax(data_dict['cluster_ref'] * pred_masks, 1) # (B,)
        pred_center = data_dict['center'] # (B,K,3)
        pred_heading_class = torch.argmax(data_dict['heading_scores'], -1) # B,num_proposal
        pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = pred_heading_class # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2) # B,num_proposal
        pred_size_class = torch.argmax(data_dict['size_scores'], -1) # B,num_proposal
        pred_size_residual = torch.gather(data_dict['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_class = pred_size_class
        pred_size_residual = pred_size_residual.squeeze(2) # B,num_proposal,3

        topk = 10
        pred_answers_top10 = data_dict['answer_scores'].topk(topk, dim=1)[1]
        pred_answer_idxs = pred_answers_top10.tolist()

        for i in range(pred_ref.shape[0]):
            # compute the iou
            pred_ref_idx = pred_ref[i]
            pred_obb = DC.param2obb(
                pred_center[i, pred_ref_idx, 0:3].detach().cpu().numpy(), 
                pred_heading_class[i, pred_ref_idx].detach().cpu().numpy(), 
                pred_heading_residual[i, pred_ref_idx].detach().cpu().numpy(),
                pred_size_class[i, pred_ref_idx].detach().cpu().numpy(), 
                pred_size_residual[i, pred_ref_idx].detach().cpu().numpy()
            )
            pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])

            # answer
            #pred_answer = dataset.answer_vocab.itos(pred_answer_idxs[i])
            pred_answers_top10 = [dataset.answer_vocab.itos(pred_answer_idx) for pred_answer_idx in pred_answer_idxs[i]]

            # store data
            scanqa_idx = data_dict["scan_idx"][i].item()
            pred_data = {
                "scene_id": scanqa[scanqa_idx]["scene_id"],
                "question_id": scanqa[scanqa_idx]["question_id"],
                "answer_top10": pred_answers_top10,
                "bbox": pred_bbox.tolist(),
            }
            pred_bboxes.append(pred_data)

    # dump
    print("dumping...")
    pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "pred."+args.test_type+".json")

    with open(pred_path, "w") as f:
        json.dump(pred_bboxes, f, indent=4)

    print("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--test_type", type=str, help="test_w_obj or test_wo_obj", default="test_wo_obj")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
    parser.add_argument("--trial", type=int, default=-1, help="trial number")    
    args = parser.parse_args()
    train_args = json.load(open(os.path.join(CONF.PATH.OUTPUT, args.folder, "info.json")))
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # overwrite
    args = overwrite_config(args, train_args)
    seed = args.seed

    # reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    predict(args)