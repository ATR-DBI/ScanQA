import time
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

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from utils.misc import overwrite_config
from lib.config import CONF
from lib.dataset import ScannetQADataset, ScannetQADatasetConfig
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from models.qa_module import ScanQA
from lib.config import CONF


project_name = "ScanQA_v1.0"
SCANQA_TRAIN = json.load(open(os.path.join(CONF.PATH.SCANQA, project_name + "_train.json"))) 


def get_dataloader(args, scanqa, all_scene_list, split, config):
    answer_vocab_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "answer_vocab.json")
    answer_counter = json.load(open(answer_vocab_path))
    answer_cands = sorted(answer_counter.keys())
    config.num_answers = len(answer_cands)

    print("using {} answers".format(config.num_answers))

    if 'bert-' in args.tokenizer_name: 
        from transformers import AutoTokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = None

    dataset = ScannetQADataset(
        scanqa=scanqa, 
        scanqa_all_scene=all_scene_list, 
        #use_unanswerable=(not args.no_unanswerable),
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
    print("evaluate on {} samples".format(len(dataset)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataset, dataloader

def get_model(args, config):
    # load tokenizer model
    if "bert-" in args.tokenizer_name:
        from transformers import AutoConfig        
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

    model_name = "model_last.pth" if args.detection else "model.pth"
    model_path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
    print('loading model from:', model_path)
    # to CUDA
    model = model.cuda()    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model


def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])
    return scene_list


def get_scanqa(args):
    if args.detection:
        scene_list = get_scannet_scene_list("val")
        scanqa = []
        for scene_id in scene_list:
            data = deepcopy(SCANQA_TRAIN[0])
            data["scene_id"] = scene_id
            scanqa.append(data)
    else:    
        SCANQA_VAL = json.load(open(os.path.join(CONF.PATH.SCANQA, project_name + "_val.json")))
        scanqa = SCANQA_TRAIN if args.use_train else SCANQA_VAL
        scene_list = sorted(list(set([data["scene_id"] for data in scanqa])))
        if args.num_scenes != -1:
            scene_list = scene_list[:args.num_scenes]

        scanqa = [data for data in scanqa if data["scene_id"] in scene_list]

    return scanqa, scene_list


def eval_qa(args):
    print("evaluate localization...")
    # constant
    DC = ScannetQADatasetConfig()

    # init training dataset
    print("preparing data...")
    scanqa, scene_list = get_scanqa(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanqa, scene_list, "val", DC)

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

    dataset = dataloader.dataset
    scanqa = dataset.scanqa
    # random seeds
    seeds = [args.seed] + [2 * i for i in range(args.repeat - 1)]

    # evaluate
    print("evaluating...")
    score_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "score.val.pkl")

    pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "pred.val.pkl")
    gen_flag = (not os.path.exists(score_path)) or args.force or args.repeat > 1
    if gen_flag:
        ref_acc_all = []
        lang_acc_all = []
        ious_all = []
        answer_acc_at1_all = []
        answer_acc_at10_all = []

        for trial, seed in enumerate(seeds):
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)

            print("generating the scores for seed {}...".format(seed))
            ref_acc = []
            lang_acc = []
            ious = []
            answer_acc_at1 = []
            answer_acc_at10 = []
            predictions = {}

            for data in tqdm(dataloader):
                # move to cuda
                for key in data:
                    if type(data[key]) is dict:
                        data[key] = {k:v.cuda() for k, v in data[key].items()}
                    else:
                        data[key] = data[key].cuda()
                # feed
                with torch.no_grad():
                    data = model(data)
                    _, data = get_loss(
                        data_dict=data, 
                        config=DC, 
                        detection=True,
                        use_reference=(not args.no_reference), 
                        use_lang_classifier=(not args.no_lang_cls),
                        use_answer=(not args.no_answer),
                    )
                    data = get_eval(
                        data_dict=data, 
                        config=DC,
                        #answer_vocab=None if (not args.no_unanswerable) else dataset.answer_vocab,
                        answer_vocab=None,
                        use_reference=True, 
                        use_lang_classifier=not args.no_lang_cls,
                        post_processing=POST_DICT
                    )
                if "ref_acc" in data:
                    ref_acc += data["ref_acc"]
                ious += data["ref_iou"]
                if "lang_acc" in data:
                    lang_acc.append(data["lang_acc"].item())                
                answer_acc_at1.append(data["answer_acc_at1"].item())
                answer_acc_at10.append(data["answer_acc_at10"].item())

                # store predictions
                ids = data["scan_idx"].detach().cpu().numpy()
                pred_answer_idxs = data["pred_answers_at10"].tolist()

                for i in range(ids.shape[0]):
                    idx = ids[i]
                    scene_id = scanqa[idx]["scene_id"]
                    question_id = scanqa[idx]["question_id"]

                    if scene_id not in predictions:
                        predictions[scene_id] = {}

                    if question_id not in predictions[scene_id]:
                        predictions[scene_id][question_id] = {}

                    predictions[scene_id][question_id]["pred_bbox"] = data["pred_bboxes"][i]
                    predictions[scene_id][question_id]["gt_bbox"] = data["gt_bboxes"][i]
                    predictions[scene_id][question_id]["iou"] = data["ref_iou"][i]

                    pred_answers_at10 = [dataset.answer_vocab.itos(pred_answer_idx) for pred_answer_idx in pred_answer_idxs[i]]
                    predictions[scene_id][question_id]["pred_answers_at10"] = pred_answers_at10

            # save the last predictions
            with open(pred_path, "wb") as f:
                pickle.dump(predictions, f)
                
            # convert pkl to json
            conved=[]
            for scene_name, scene in predictions.items():
                for qid, instance in scene.items():
                    instance = {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in instance.items()}
                    instance.update({'scene_id': scene_name, 'question_id': qid})
                    instance['answer_top10'] = instance['pred_answers_at10']
                    del instance['pred_answers_at10']
                    instance['bbox'] = instance['pred_bbox']
                    del instance['pred_bbox']
                    conved.append(instance)
            json.dump(conved,open(pred_path[:-4]+'.json','w'))

            # save to global
            ref_acc_all.append(ref_acc)
            lang_acc_all.append(lang_acc)
            ious_all.append(ious)
            answer_acc_at1_all.append(answer_acc_at1)
            answer_acc_at10_all.append(answer_acc_at10)

        # convert to numpy array
        ref_acc = np.array(ref_acc_all)
        lang_acc = np.array(lang_acc_all)
        answer_acc_at1  = np.array(answer_acc_at1_all) 
        answer_acc_at10 = np.array(answer_acc_at10_all) 
        ious = np.array(ious_all)

        # save the global scores
        with open(score_path, "wb") as f:
            scores = {
                "ref_acc": ref_acc_all,
                "lang_acc": lang_acc_all,
                "answer_acc_at1": answer_acc_at1_all,
                "answer_acc_at10": answer_acc_at10_all,
                "ious": ious_all,
            }
            pickle.dump(scores, f)

    else:
        print("loading the scores...")
        with open(score_path, "rb") as f:
            scores = pickle.load(f)
            # unpack
            ref_acc = np.array(scores["ref_acc"])
            lang_acc = np.array(scores["lang_acc"])
            ious = np.array(scores["ious"])
            answer_acc_at1  = np.array(scores["answer_acc_at1"])
            answer_acc_at10 = np.array(scores["answer_acc_at10"])

    if len(lang_acc) != 0:
        print("\nlanguage classification accuracy: {}".format(np.mean(lang_acc)))
    # ACCURACY for only answerable questions, not all questions
    # print("\n[answerbele] answer accuracy @1: {}, @10: {}".format(np.mean(answer_acc_at1), np.mean(answer_acc_at10)))


def eval_det(args):
    print("evaluate detection...")
    # constant
    DC = ScannetQADatasetConfig()
    
    # init training dataset
    print("preparing data...")
    scanqa, scene_list = get_scanqa(args)

    # dataloader
    dataset, dataloader = get_dataloader(args, scanqa, scene_list, "val", DC)
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
    }
    AP_IOU_THRESHOLDS = [0.25, 0.5]
    AP_CALCULATOR_LIST = [APCalculator(iou_thresh, DC.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

    sem_acc = []
    for data in tqdm(dataloader):
        for key in data:
            data[key] = data[key].cuda()

        # feed
        with torch.no_grad():
            data = model(data)
            _, data = get_loss(
                data_dict=data, 
                config=DC, 
                detection=True,
                reference=False
            )
            data = get_eval(
                data_dict=data, 
                config=DC, 
                reference=False,
                post_processing=POST_DICT
            )

        sem_acc.append(data["sem_acc"].item())

        batch_pred_map_cls = parse_predictions(data, POST_DICT) 
        batch_gt_map_cls = parse_groundtruths(data, POST_DICT) 
        for ap_calculator in AP_CALCULATOR_LIST:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # aggregate object detection results and report
    print("\nobject detection sem_acc: {}".format(np.mean(sem_acc)))
    for i, ap_calculator in enumerate(AP_CALCULATOR_LIST):
        print()
        print("-"*10, "iou_thresh: %f"%(AP_IOU_THRESHOLDS[i]), "-"*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            print("eval %s: %f"%(key, metrics_dict[key]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model", required=True)
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--force", action="store_true", help="enforce the generation of results")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times for evaluation")
    #parser.add_argument("--no_unanswerable", action="store_true", help="Do'not use unanswerable examples")
    parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
    parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.") 
    parser.add_argument("--qa", action="store_true", help="evaluate the qa results")
    parser.add_argument("--detection", action="store_true", help="evaluate the object detection results")
    args = parser.parse_args()
    train_args = json.load(open(os.path.join(CONF.PATH.OUTPUT, args.folder, "info.json")))

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # overwrite
    args = overwrite_config(args, train_args)

    # evaluate
    if args.qa: eval_qa(args)
    if args.detection: eval_det(args)

