from time import sleep
import copy
from collections import Counter, defaultdict
import re,glob,csv,json
import sys,os
import pickle

import nltk
nltk.download('omw-1.4')

import random
import numpy as np
from collections import defaultdict
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

sys.path.append(os.path.join(os.getcwd()))
from lib.config import CONF


def get_lemma(ss):
    return [lemmatizer.lemmatize(token) for token in ss.split()]


def simple_ratio(numerator,denominator): 
    num_numerator=sum([1 if token in numerator else 0 for token in denominator])
    num_denominator=len(denominator)
    return num_numerator/num_denominator


def tokens_unigram_f_value(ref: str,pred: str)->float:
    ref_lemma = get_lemma(ref)
    pred_lemma = get_lemma(pred)
    precision = simple_ratio(ref_lemma,pred_lemma)
    recall    = simple_ratio(pred_lemma,ref_lemma)
    return 2*(recall*precision)/(recall+precision) if recall+precision!=0. else 0


def tokens_score(ref: str,pred: str)->float:
    return 1. if ref==pred else 0.


def evals_json(gold_data,preds):
    score_list = ['Top1 (EM)','Top10 (EM)','Top1 (F-value)']
    score = {s:[] for s in score_list}
    
    for ins in gold_data:
        question_id=ins['question_id']
        question=ins['question']
        ref_answers=ins['answers']
        scene_id=ins['scene_id']
        pred=preds[question_id]

        # top-1
        answer = pred['answer_top10'][0]
        if answer in ref_answers:
            score['Top1 (EM)'].append(1)
            score['Top1 (F-value)'].append(1)
        else:
            scores=[tokens_unigram_f_value(answer,ref) for ref in ref_answers]
            score['Top1 (EM)'].append(0)
            score['Top1 (F-value)'].append(max(scores))

        # top-10
        for answer in pred['answer_top10']:
            if answer in ref_answers:
                score['Top10 (EM)'].append(1)
                break
        else:
            score['Top10 (EM)'].append(0)
        
    rlt={}
    for k,v in score.items():
        assert len(v)==len(gold_data),len(v)
        print(k,np.mean(v)*100)
        rlt[k]=np.mean(v)*100
    return rlt

def eval_pycoco(gold_data, preds, use_spice=False):
    score_list = ['Top1 (EM)','Top10 (EM)','Top1 (F-value)','BLEU-1','BLEU-2','BLEU-3','BLEU-4']
    score = {s:[] for s in score_list}
    
    scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
    ]
    if use_spice:
        scorers.append((Spice(), "SPICE"))

    tokenizer = PTBTokenizer()
    # pycocoeval
    gts = {ins['question_id']:[{'caption':ans} for ans in ins['answers']] for ins in gold_data}
    res = {qid:[{'caption':value['answer_top10'][0]}] for qid,value in preds.items()}
    gts  = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    #print(gts,res)
    
    # =================================================
    # Compute scores
    # =================================================
    rlt={}
    for scorer, method in scorers:
        eprint('computing %s score...'%(scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                print("%s: %0.3f"%(m, sc*100))
                rlt[m]=sc*100
        else:
            print("%s: %0.3f"%(method, score*100))
            rlt[method]=score*100
    return rlt

QT=['Place','Number','Color','Object nature','Object','Other']
def qclass1(question):
    lques = question
    if 'Where' in lques:
        return 'Place'
    if 'How many' in lques:
        return 'Number'
    if 'What color' in lques or 'What is the color' in lques:
        return 'Color'
    if 'What shape' in lques:
        #return 'Shape'
        return 'Object nature'
    if 'What type' in lques:
        #return 'Type'
        return 'Object nature'
    if 'What kind' in lques:
        #return 'Kind'
        return 'Object nature'
    if 'What is' in lques:
        return 'Object'
    return 'Other'
            
if __name__=="__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--folder", type=str, help="Folder containing the results", required=True)
    parser.add_argument('--use_spice',  help='no spice', action="store_true")
    parser.add_argument('--detailed', help='', action="store_true")
    args = parser.parse_args()

    SPLITS=['val']
    
    ds={split:json.load(open(os.path.join(CONF.PATH.SCANQA, f'ScanQA_v1.0_{split}.json'))) for split in SPLITS}

    if args.detailed:
        for split in  ['val','test_w_obj','test_wo_obj']:
                
            if split in ['val','test_w_obj']:
                fin=open(os.path.join(CONF.PATH.OUTPUT, args.folder, f'predictions.{split}.p'))
                preds=pickle.load(open(fin,'rb'))
                preds={qid:value for scene,pred_scenes in preds.items() for qid,value in pred_scenes.items()} # simplify data format to dict[qid,pred]
                for _,pred in preds.items():
                    assert 'answer_top10' not in pred
                    pred['answer_top10']=pred['pred_answers_at10']
                    del pred['pred_answers_at10']
                print("# Loaded",fin,len(preds))
            else:
                fin=open(os.path.join(CONF.PATH.OUTPUT, args.folder, 'pred.json'))
                preds=json.load(open(fin,'r'))
                print("# Loaded",fin,len(preds))
                preds={q['question_id']:q for q in preds}  # simplify data format to dict[qid,pred]

            golds=ds[split]
            scores={}

            preds_={k:{} for k in QT}
            golds_={k:[] for k in QT}
            #for qid,g in golds.items():
            for g in golds:
                qid=g['question_id']
                preds_[qclass1(g['question'])][qid]=preds[qid]
                golds_[qclass1(g['question'])].append(g)

            for qt in QT:
                score=evals_json(golds_[qt],preds_[qt])
                #print()
                score2=eval_pycoco(golds_[qt], preds_[qt], use_spice=args.use_spice)
                score.update(score2)
                scores[f"{split}.{qt}"]=score
            print(split,scores)
            json.dump(scores,open(fin+'.eval.detailed.json','w'),indent=4,sort_keys=True)
            print()
            print()
        quit()

    #
    # val
    #
    fin=os.path.join(CONF.PATH.OUTPUT, args.folder, 'pred.val.pkl')
    preds=pickle.load(open(fin,'rb'))
    preds={qid:value for scene,pred_scenes in preds.items() for qid,value in pred_scenes.items()} # simplify data format to dict[qid,pred]
    for _,pred in preds.items():
        pred['answer_top10']=pred['pred_answers_at10']
        del pred['pred_answers_at10']
    print("# Loaded",fin,len(preds))
    
    score=evals_json(ds['val'],preds)
    #print()
    eval_pycoco(ds['val'], preds, use_spice=args.use_spice)
    print()
    print()
