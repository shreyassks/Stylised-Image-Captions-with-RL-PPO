import sys
sys.path.append("cider")
sys.path.append("coco_caption")

from pyciderevalcap.ciderD.ciderD import CiderD
from pyciderevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.bleu_1.bleu import Bleu
from coco_caption.pycocoevalcap.bleu.bleu import Bleu as blah

import numpy as np
from collections import OrderedDict

CiderD_scorer = None
Cider_scorer = None
Bleu_scorer = None
Blah_scorer = None

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Cider_scorer
    Cider_scorer = Cider_scorer or Cider(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)
    global Blah_scorer
    Blah_scorer = Blah_scorer or blah(4)


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def get_self_critical_reward(rank, model, fc_feats, att_feats, densecap, personality, ground_ts, gen_result, opt):
    batch_size = gen_result.size(0)
    seq_per_img = batch_size // len(ground_ts)

    # get greedy decoding baseline
    greedy_res, _, _ = model.module._sample(rank, fc_feats, att_feats, densecap, att_masks=None, 
                                            personality=personality, opt={"sample_method":"greedy"})

    res = OrderedDict()
    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    ground_ts = ground_ts.squeeze(1).cpu().numpy()

    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(ground_ts)):
        gts[i] = [array_to_str(ground_ts[i])]

    res = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res)

    else:
        cider_scores = 0

    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res)
        bleu_scores = np.array(bleu_scores[3])
        # print('Bleu scores:', _[3])
    else:
        bleu_scores = 0

    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores
    # scores = scores[:batch_size] - beta*scores[batch_size:]
    return scores


def cal_cider(data_gts, gen_result):
    batch_size = 1
    res = OrderedDict()
    for i in range(batch_size):
        res[i] = [gen_result]
    gts = OrderedDict()
    for i in range(batch_size):
        gts[i] = data_gts

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(batch_size)]
    _, cider_scores = CiderD_scorer.compute_score(gts, res_)
    return _ 


def get_scores_separate(data_gts, gen_result):
    allscore ={}
    batch_size = len(gen_result)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size//len(data_gts)

    res = OrderedDict()
    
    #gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [gen_result[i]]
    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = data_gts[i]
        #gts[i] = [data_gts[i][j] for j in range(len(data_gts[i]))]
    res_ = [{'image_id':i, 'caption': res[i]} for i in range(batch_size)]
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i // seq_per_img] for i in range(batch_size)}
    cider_score, cider_scores = Cider_scorer.compute_score(gts, res_)
    #print('Cider score:', cider_score)
    allscore['Cider'] = cider_score
    bleu_score, bleu_scores = Blah_scorer.compute_score(gts, res__)
    for index, b in enumerate(bleu_score):
        allscore["Bleu"+str(index+1)]=b
    bleu_scores = np.array(bleu_scores[3])
    #print('Bleu scores:', _[3])
    return allscore 
