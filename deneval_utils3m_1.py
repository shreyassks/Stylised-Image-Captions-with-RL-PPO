from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import sys
sys.path.append("coco_caption")
    
import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
    
import misc.utils as utils
from misc.rewards import init_scorer, cal_cider, get_scores_separate
import pandas as pd
import numpy as np
import random

rank = torch.device("cuda")

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0

def language_eval(dataset, preds, model_id, split):
    try:
        if not os.path.isdir('eval_results'):
            os.mkdir('eval_results')

        cache_path = os.path.join('eval_results/', model_id + '_' + split + '_1.json')
        best_cider=0

        gdindex=[-1]
        cider_list =[]
        for i in gdindex:
            annFile='coco_caption/person_captions4eval_'+str(i)+'.json'
            print(annFile)
            coco = COCO(annFile)    
            valids = coco.getImgIds()

            # filter results to only those in MSCOCO validation set (will be about a third)
            preds_filt = [p for p in preds if p['image_id'] in valids]
            print('using %d/%d predictions' % (len(preds_filt), len(preds)))
            json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

            cocoRes = coco.loadRes(cache_path)
            cocoEval = COCOEvalCap(coco, cocoRes)
            cocoEval.params['image_id'] = cocoRes.getImgIds()
            cocoEval.evaluate()
            cider_list.append(cocoEval.eval['CIDEr'])

            # create output dictionary
            if cocoEval.eval['CIDEr']>=best_cider:
                best_cider = cocoEval.eval['CIDEr']
                out = {}
                for metric, score in cocoEval.eval.items():
                    out[metric] = score

                imgToEval = cocoEval.imgToEval

                for p in preds_filt:
                    image_id, caption = p['image_id'], p['caption']
                    imgToEval[image_id]['caption'] = caption
                #update predictions
                for i in range(len(preds)):
                    if preds[i]['image_id'] in imgToEval:
                        preds[i]['eval'] = imgToEval[preds[i]['image_id']]

                out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
            else:
                continue

        outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

        with open(outfile_path, 'w') as outfile:
            c = {'overall': out, 'imgToEval': imgToEval}
            json.dump([preds_filt, c], outfile)
        
        with open(outfile_path, 'w') as outfile:
            c = {'overall': out, 'imgToEval': imgToEval}
            json.dump([c], outfile)

        cider_list=np.array(cider_list)
        print("min:",np.min(cider_list)," max:",np.max(cider_list)," mean:",np.mean(cider_list)," std:",np.std(cider_list))
        return out
    
    except json.decoder.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    rank_eval = eval_kwargs.get('rank_eval', 0)
    dataset = eval_kwargs.get('dataset', 'person')
    beam_size = eval_kwargs.get('beam_size', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings) # Use this nasty way to make other code clean since it's a global configuration
    use_joint=eval_kwargs.get('use_joint', 0)
    init_scorer('cider_words/person-'+split+'-words')
    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    losses={}
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    visual={"image_id":[],"personality":[],"generation":[],"gd":[],"densecap":[],"Bleu1_gen/cap":[],
            "Bleu2_gen/cap":[],"Bleu3_gen/cap":[],"Bleu4_gen/cap":[],"Cider_gen/cap":[],"Bleu1_gen/gd":[],
            "Bleu2_gen/gd":[],"Bleu3_gen/gd":[],"Bleu4_gen/gd":[],"Cider_gen/gd":[],"Bleu1_cap/gd":[],
            "Bleu2_cap/gd":[],"Bleu3_cap/gd":[],"Bleu4_cap/gd":[],"Cider_cap/gd":[], "Bleu1_gd/gen":[],
            "Bleu2_gd/gen":[],"Bleu3_gd/gen":[],"Bleu4_gd/gen":[],"Cider_gd/gen":[]}

    minopt=0
    verbose_loss = True
    
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size)], 
            data['att_feats'][np.arange(loader.batch_size)] if data['att_feats'] is not None else None,
            data['densecap'][np.arange(loader.batch_size)],
            data['att_masks'][np.arange(loader.batch_size)] if data['att_masks'] is not None else None,
            data['personality'][np.arange(loader.batch_size)]]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats,densecap, att_masks,personality = tmp
        ground_truth =  data['labels'][:][:,1:]
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model._sample(rank, fc_feats, att_feats,densecap, att_masks, personality, opt=eval_kwargs)[0].data
        
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        gd_display = utils.decode_sequence(loader.get_vocab(), ground_truth)
        for k, s in enumerate(sents):
            if beam_size > 1 and verbose_beam:
                beam_sents = [utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[k]] 
                maxcider=0
                mincider=1000
                sent =s
                for b,sq in enumerate(beam_sents):
                    current_cider=cal_cider(gd_display[k*loader.seq_per_img:(k+1)*loader.seq_per_img],sq)
                    if current_cider >= maxcider:
                        maxcider=current_cider
                        sentmax=sq
                    if current_cider <= mincider:
                        mincider=current_cider
                        sentmin=sq
                    if minopt==1:
                        sent=sentmin
                    elif minopt==-1:
                        sent=sentmax
                    else:
                        sent=s
                    
            else:
                sent = s
            #print("best sentence: ",sent) 
            newpidstr = str(personality[k].nonzero()[0].item())
            changed_personality =loader.get_personality()[newpidstr]
            entry = {'image_id': data['infos'][k]['id']+"_"+data['infos'][k]['personality'], 
                     'caption':sent,'gd':gd_display[k*loader.seq_per_img:(k+1)*loader.seq_per_img]}
            
            if( entry not in predictions ):
                densecap_display = utils.decode_sequence(loader.get_vocab(), data['densecap'][k])
                allscore = get_scores_separate([densecap_display],[sent]) # gd is the densecap and test is generation, len(common)/len(generation)
                for bk in allscore:
                    visual[bk+"_gen/cap"].append(allscore[bk])
                allscore_gd = get_scores_separate([gd_display[k*loader.seq_per_img:(k+1)*loader.seq_per_img]],[sent])
                for bkgd in allscore_gd:
                    visual[bkgd+"_gen/gd"].append(allscore_gd[bkgd])
                allscore_capgd = get_scores_separate([gd_display[k*loader.seq_per_img:(k+1)*loader.seq_per_img]],densecap_display)
                for cap_bkgd in allscore_capgd:
                    visual[cap_bkgd+"_cap/gd"].append(allscore_capgd[cap_bkgd])
                
                allscore_gd_flip = get_scores_separate([[sent]],gd_display[k*loader.seq_per_img:(k+1)*loader.seq_per_img]) 
                for bkgd in allscore_gd_flip:
                    visual[bkgd+"_gd/gen"].append(allscore_gd_flip[bkgd])                
                
                visual["image_id"].append(data['infos'][k]['id'])
                visual["personality"].append(data['infos'][k]['personality'])
                if split=='change':
                    visual["new_personality"].append(changed_personality)
                visual['generation'].append(sent)
                visual["gd"].append(gd_display[k*loader.seq_per_img:(k+1)*loader.seq_per_img])
                visual["densecap"].append(densecap_display)
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)

            if verbose:
                print('--------------------------------------------------------------------')
                if split=='change':
                    print('image %s{%s--------->%s}: %s' %(entry['image_id'],changed_personality,entry['gd'], entry['caption']))
                else:
                    print('image %s{%s}: %s' %(entry['image_id'],entry['gd'], entry['caption']))
                print('--------------------------------------------------------------------')

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
            
        for i in range(n - ix1):
            predictions.pop()
        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))
        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break
            
    allwords = " ".join(visual['generation'])
    allwords = allwords.split(" ")
    
    print("sets length of allwords:",len(set(allwords)))
    print("length of allwords:",len(allwords))
    print("rate of set/all:",len(set(allwords))/len(allwords))
    
    with open("data/predictions.json", 'w') as outfile:
        json.dump(predictions, outfile)
    
    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)
    
    # df = pd.DataFrame.from_dict(visual)
    # df.to_csv("visual_res/"+eval_kwargs['id']+"_"+str(split)+"_"+"visual.csv")
    
    return predictions, lang_stats