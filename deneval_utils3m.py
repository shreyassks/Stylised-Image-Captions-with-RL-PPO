from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import torch
import numpy as np
import json

import misc.utils as utils
from misc.rewards import init_scorer, cal_cider, get_scores_separate

from coco_caption.pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
sys.path.append("coco_caption")

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']


def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


def language_eval(preds, model_id, split):
    try:
        if not os.path.isdir('eval_results'):
            os.mkdir('eval_results')

        cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
        best_cider = 0

        gdindex = [-1]
        cider_list = []

        for i in gdindex:
            annFile = 'coco_caption/person_captions4eval_'+str(i)+'.json'
            coco = COCO(annFile)
            valids = coco.getImgIds()

            preds_filt = [p for p in preds if p['image_id'] in valids]
            print('using %d/%d predictions' % (len(preds_filt), len(preds)))
            json.dump(preds_filt, open(cache_path, 'w'))

            cocoRes = coco.loadRes(cache_path)
            cocoEval = COCOEvalCap(coco, cocoRes)
            cocoEval.params['image_id'] = cocoRes.getImgIds()
            cocoEval.evaluate()

            cider_list.append(cocoEval.eval['CIDEr'])
            # create output dictionary
            if cocoEval.eval['CIDEr'] >= best_cider:
                best_cider = cocoEval.eval['CIDEr']
                out = {}
                for metric, score in cocoEval.eval.items():
                    out[metric] = score

                imgToEval = cocoEval.imgToEval

                for p in preds_filt:
                    image_id, caption = p['image_id'], p['caption']
                    imgToEval[image_id]['caption'] = caption

                for i in range(len(preds)):
                    if preds[i]['image_id'] in imgToEval:
                        preds[i]['eval'] = imgToEval[preds[i]['image_id']]

                out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
            else:
                continue

        with open(cache_path, 'w') as outfile:
            c = {'overall': out, 'imgToEval': imgToEval}
            json.dump(c, outfile)

        cider_list = np.array(cider_list)
        print("min:", np.min(cider_list), " max:", np.max(cider_list), " mean:",np.mean(cider_list), " std:", np.std(cider_list))
        return out

    except json.decoder.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")


def eval_split(rank, model, crit, loader, ds, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    beam_size = eval_kwargs.get('beam_size', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings)
    init_scorer('cider_words/person-'+split+'-words')

    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    mu = 0.9
    visual = {"image_id":[],"personality":[],"generation":[],"gd":[],"densecap":[],"Bleu1_gen/cap":[],"Bleu2_gen/cap":[],
            "Bleu3_gen/cap":[],"Bleu4_gen/cap":[],"Cider_gen/cap":[],"Bleu1_gen/gd":[],"Bleu2_gen/gd":[],"Bleu3_gen/gd":[],
            "Bleu4_gen/gd":[],"Cider_gen/gd":[],"Bleu1_cap/gd":[],"Bleu2_cap/gd":[],"Bleu3_cap/gd":[],"Bleu4_cap/gd":[],
            "Cider_cap/gd":[], "Bleu1_gd/gen":[],"Bleu2_gd/gen":[],"Bleu3_gd/gen":[],"Bleu4_gd/gen":[],"Cider_gd/gen":[]}

    minopt = 0
    for i, (fc_feats, att_feats, densecap, seq_labels, gts, seq_masks, personality, target2, idx, infos) in enumerate(loader):
        tmp = [fc_feats, att_feats, densecap, seq_labels, gts, seq_masks, personality, target2]
        tmp = [i if i is None else i.to(rank, non_blocking=True) for i in tmp]

        fc_feats, att_feats, densecap, seq_labels, gts, seq_masks, personality, target2 = tmp
        att_masks = torch.ones(att_feats.size(0), 7*7, dtype=torch.int64)

        with torch.no_grad():
            outs1, outs2 = model(rank, fc_feats, att_feats, densecap, seq_labels, att_masks, personality)
            loss1, loss2 = crit(outs1, seq_labels[:, 1:], seq_masks[:, 1:], outs2, target2)
            loss = mu*loss1 + (1-mu)*loss2

            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        ground_truth = seq_labels[:, 1:]

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model.module._sample(rank, fc_feats, att_feats, densecap, att_masks, personality, opt=eval_kwargs)[0].data

        sents = utils.decode_sequence(ds.get_vocab(), seq)
        gd_display = utils.decode_sequence(ds.get_vocab(), ground_truth)

        for k, s in enumerate(sents):
            if beam_size > 1 and verbose_beam:
                beam_sents = [utils.decode_sequence(ds.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[k]] 
                maxcider = 0
                mincider = 1000
                sent = s
                for b, sq in enumerate(beam_sents):
                    current_cider = cal_cider(gd_display[k*ds.seq_per_img:(k+1) * ds.seq_per_img], sq)
                    if current_cider >= maxcider:
                        maxcider = current_cider
                        sentmax = sq
                    if current_cider <= mincider:
                        mincider = current_cider
                        sentmin = sq
                    if minopt == 1:
                        sent = sentmin
                    elif minopt == -1:
                        sent = sentmax
                    else:
                        sent = s
            else:
                sent = s

            entry = {'image_id': infos['id'][k]+"_"+infos['personality'][k], 
                     'caption': sent, 'gd': gd_display[k*ds.seq_per_img:(k+1)*ds.seq_per_img]}

            if (entry not in predictions):
                densecap_display = utils.decode_sequence(ds.get_vocab(), densecap[k])
                allscore = get_scores_separate([densecap_display],[sent])
                for bk in allscore:
                    visual[bk+"_gen/cap"].append(allscore[bk])
                allscore_gd = get_scores_separate([gd_display[k*ds.seq_per_img:(k+1)*ds.seq_per_img]],[sent])
                for bkgd in allscore_gd:
                    visual[bkgd+"_gen/gd"].append(allscore_gd[bkgd])
                allscore_capgd = get_scores_separate([gd_display[k*ds.seq_per_img:(k+1)*ds.seq_per_img]],densecap_display)
                for cap_bkgd in allscore_capgd:
                    visual[cap_bkgd+"_cap/gd"].append(allscore_capgd[cap_bkgd])

                allscore_gd_flip = get_scores_separate([[sent]],gd_display[k*ds.seq_per_img:(k+1)*ds.seq_per_img])
                for bkgd in allscore_gd_flip:
                    visual[bkgd+"_gd/gen"].append(allscore_gd_flip[bkgd])

                visual["image_id"].append(infos['id'][k])
                visual["personality"].append(infos['personality'][k])
                visual['generation'].append(sent)
                visual["gd"].append(gd_display[k*ds.seq_per_img:(k+1)*ds.seq_per_img])
                visual["densecap"].append(densecap_display)

            predictions.append(entry)

            if verbose:
                print('--------------------------------------------------------------------')
                print('image %s{%s}: %s' %(entry['image_id'], entry['gd'], entry['caption']))
                print('--------------------------------------------------------------------')

    allwords = " ".join(visual['generation'])
    allwords = allwords.split(" ")

    print("sets length of allwords:",len(set(allwords)))
    print("length of allwords:",len(allwords))
    print("rate of set/all:",len(set(allwords))/len(allwords))

    lang_stats = None
    if lang_eval == 1:
        print("Language Evaluation")
        lang_stats = language_eval(predictions, eval_kwargs['id'], split)

    val_loss = loss_sum/loss_evals
    print(f"Validation Loss: {val_loss}")

    return val_loss, predictions, lang_stats


def encode_data(model, loader, eval_kwargs={}):
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode
    model.eval()

    loader_seq_per_img = loader.seq_per_img
    loader.seq_per_img = 5
    loader.reset_iterator(split)

    n = 0
    img_embs = []
    cap_embs = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        tmp = utils.var_wrapper(tmp)
        fc_feats, att_feats, labels, masks = tmp

        with torch.no_grad():
            img_emb = model.vse.img_enc(fc_feats)
            cap_emb = model.vse.txt_enc(labels, masks)

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)

        if n > ix1:
            img_emb = img_emb[:(ix1-n)*loader.seq_per_img]
            cap_emb = cap_emb[:(ix1-n)*loader.seq_per_img]

        # preserve the embeddings by copying from gpu and converting to np
        img_embs.append(img_emb.data.cpu().numpy().copy())
        cap_embs.append(cap_emb.data.cpu().numpy().copy())

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

        print("%d/%d"%(n,ix1))

    img_embs = np.vstack(img_embs)
    cap_embs = np.vstack(cap_embs)

    assert img_embs.shape[0] == ix1 * loader.seq_per_img

    loader.seq_per_img = loader_seq_per_img

    return img_embs, cap_embs


def evalrank(model, loader, eval_kwargs={}):
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')
    fold5 = eval_kwargs.get('fold5', 0)
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    print('Computing results...')
    img_embs, cap_embs = encode_data(model, loader, eval_kwargs)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        r, rt = i2t(img_embs, cap_embs, measure='cosine', return_ranks=True)
        ri, rti = t2i(img_embs, cap_embs,
                      measure='cosine', return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000],
                         cap_embs[i * 5000:(i + 1) *
                                  5000], measure='cosine',
                         return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],
                           cap_embs[i * 5000:(i + 1) *
                                    5000], measure='cosine',
                           return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    return {'rsum':rsum, 'i2t_ar':ar, 't2i_ar':ari,
            'i2t_r1':r[0], 'i2t_r5':r[1], 'i2t_r10':r[2], 'i2t_medr':r[3], 'i2t_meanr':r[4],
            't2i_r1':ri[0], 't2i_r5':ri[1], 't2i_r10':ri[2], 't2i_medr':ri[3], 't2i_meanr':ri[4]}#{'rt': rt, 'rti': rti}


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    ims = np.array([images[i] for i in range(0, len(images), 5)])

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
