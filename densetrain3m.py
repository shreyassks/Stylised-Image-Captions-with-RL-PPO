from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import traceback
import numpy as np

import opts
import models
import misc.utils as utils
from denseloader3m import YFCC_3M
import deneval_utils3m as eval_utils
from misc.rewards import get_self_critical_reward, init_scorer

import torch
from torch.autograd import Variable
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()


def save_checkpoint(opt, model, infos, optimizer, histories=None, append=''):
    if len(append) > 0:
        append = '-' + append
    # if checkpoint_path doesn't exist
    if not os.path.isdir(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    checkpoint_path = os.path.join(opt.checkpoint_path, 'model%s.pth' %(append))
    torch.save(model.state_dict(), checkpoint_path)

    print("model saved to {}".format(checkpoint_path))
    optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pth' %(append))
    torch.save(optimizer.state_dict(), optimizer_path)

    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
        utils.pickle_dump(infos, f)

    if histories:
        with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
            utils.pickle_dump(histories, f)
            

def normalize_baseline(score, batch_size):
    sample_s, greedy_s = np.mean(score[:batch_size]), np.mean(score[batch_size:])
    final_score = sample_s/greedy_s
    return final_score


def train(rank, world_size, opt):
    setup(rank, world_size)
    # Deal with feature things before anything
    opt.use_fc, opt.use_att = utils.if_use_feat(opt.caption_model)

    train_dataset = YFCC_3M(opt)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, 
                              num_workers=16, pin_memory=True, sampler=train_sampler, drop_last=True)

    print(f"Length of Train Loader : {len(train_loader)}")

    opt.input_json = opt.input_json.replace("training", "validation")
    opt.densecap_dir = opt.densecap_dir.replace("training", "validation")
    opt.input_label_h5 = opt.input_label_h5.replace("training", "validation")
    opt.input_label_start_idx = opt.input_label_start_idx.replace("training", "validation")
    opt.input_label_end_idx = opt.input_label_end_idx.replace("training", "validation")
    opt.perss_onehot_h5 = opt.perss_onehot_h5.replace("training", "validation")

    val_dataset = YFCC_3M(opt)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, 
                          num_workers=24, pin_memory=True, sampler=val_sampler, drop_last=True)

    print(f"Length of Val Loader : {len(val_loader)}")

    opt.vocab_size = train_dataset.vocab_size
    opt.seq_length = train_dataset.seq_length

    infos = {}
    histories = {}
    if opt.start_from is not None:
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'-best.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'-best.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'-best.pkl'), 'rb') as f:
                histories = utils.pickle_load(f)
    else:
        infos['iter'] = 0
        infos['epoch'] = 0
        infos['vocab'] = train_dataset.get_vocab()
        infos['pix_perss'] = train_dataset.get_personality()

    infos['opt'] = opt
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    print(f"Epoch resumed at : {epoch}")

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})
    print(f" LR History : {lr_history}")

    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    opt.vocab = train_dataset.get_vocab()
    opt.xpersonality = train_dataset.get_personality()

    model = models.setup(opt).cuda()

    del opt.vocab
    if opt.start_from is not None:
        opt.model = os.path.join(opt.start_from, 'model'+'-best.pth')
        load_model = torch.load(opt.model, map_location="cuda")
        model.load_state_dict(load_model)
        del load_model

    dp_model = DDP(model, device_ids=[rank])

    criterion = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion() if not opt.ppo else utils.PPOCriterion(opt.ppo_clip_param)

    optimizer = utils.build_optimizer([p for p in model.parameters() if p.requires_grad], opt)

    # Load the Optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer-best.pth")):
        load_opt = torch.load(os.path.join(opt.start_from, 'optimizer-best.pth'), map_location="cpu")
        optimizer.load_state_dict(load_opt)
        del load_opt
    else:
        print('Optimizer param group number not matched? There must be new parameters. Reinit the optimizer.')

    try:
        """
        set_detect_anomaly is a debugging tool to find issues with model's gradients during backpropagation. 
        When enabled, it will raise an error if a gradient computation results in NaN or infinite values.
        """
        if opt.ppo and opt.drop_prob_lm > 0:
            opt.drop_prob_lm = 0
            print('===== Highly recommend setting dropout prob to 0 during PPO training =====')
        mu, beta = 0.9, 0.8
        torch.autograd.set_detect_anomaly(True)
        num_epochs = opt.max_epochs
        trainSteps = len(train_dataset) // opt.batch_size // world_size
        opt.save_checkpoint_every = trainSteps - 1
        opt.losses_log_every = trainSteps - 1
        print_steps = opt.save_checkpoint_every // 4

        print(f"Saves Checkpoint on {opt.save_checkpoint_every} steps")
        print(f"Training Steps : {trainSteps}")
        print(f"Print Steps: {print_steps}")

        for epoch in range(epoch+1, num_epochs):
            dp_model.train()
            train_loss = 0.

            # If start self critical training (SCST)
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                opt.language_eval = 1
                init_scorer(opt.cached_tokens)
                print("SCST Training Started!! Sit back and Relax meanwhile")
            else:
                print("XE Loss Training!! Wait for some time please.")
                sc_flag = False

            # Learning Rate Decay
            if not opt.noamopt and not opt.reduce_on_plateau and not sc_flag:
                # Assign the learning rate
                if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay_rate  ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = 3e-6  # opt.learning_rate
                print(f"Setting Fixed LR to: {opt.current_lr}")
                # set the decayed rate
                utils.set_lr(optimizer, opt.current_lr)

            # Assign the scheduled sampling probability
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # Assign retrieval loss weight
            if epoch > opt.retrieval_reward_weight_decay_start and opt.retrieval_reward_weight_decay_start >= 0:
                frac = (epoch - opt.retrieval_reward_weight_decay_start) // opt.retrieval_reward_weight_decay_every
                model.retrieval_reward_weight = opt.retrieval_reward_weight * (opt.retrieval_reward_weight_decay_rate  ** frac)

            running_avg = 0.
            store_running_avg = []
            for i, (fc_feats, att_feats, densecap, seq_labels, gts, seq_masks, personality, target2, idx, infos) in enumerate(train_loader):
                tmp = [fc_feats, att_feats, densecap, seq_labels, gts, seq_masks, personality, target2]
                tmp = [i if i is None else i.to(rank, non_blocking=True) for i in tmp]
                fc_feats, att_feats, densecap, seq_labels, gts, seq_masks, personality, target2 = tmp
                att_masks = torch.ones(opt.batch_size, 7*7, dtype=torch.int64)

                # PPO self-critical update
                if opt.ppo and sc_flag:
                    gen_result, old_logprobs, _ = dp_model.module._sample(rank, fc_feats, att_feats, densecap, 
                                                                personality=personality, opt={'sample_method': "sample"})

                    old_logprobs = old_logprobs.detach()
                    old_logprobs[:, 1:] = old_logprobs[:, 1:] * Variable((gen_result[:, :-1] > 0).float(), requires_grad=False)
                    old_logprobs_agg = old_logprobs.sum(dim=1)

                    reward = get_self_critical_reward(rank, dp_model, fc_feats, att_feats, densecap, personality, gts, gen_result, opt)
                    batch_avg = normalize_baseline(reward, opt.batch_size)
                    running_avg = (i * running_avg + batch_avg) / (i + 1)
                    r_scores = reward[:opt.batch_size] - beta*running_avg*reward[opt.batch_size:]
                    # print(f"Reward Obtained is : {reward.mean()}")
                    for ppo_iter in range(opt.ppo_iters):
                        new_logprobs = dp_model.module.get_seq_logprobs(rank, fc_feats, att_feats, densecap, gen_result, None, personality)
                        new_logprobs_agg = new_logprobs.sum(dim=1)
                        loss = rl_crit(old_logprobs_agg, new_logprobs_agg, gen_result, 
                                       Variable(torch.from_numpy(r_scores).float().to(rank), requires_grad=False))
                        # print(f"Loss : {loss}")

                        optimizer.zero_grad()
                        if ppo_iter < opt.ppo_iters - 1:
                            loss.backward(retain_graph=True)
                        else:
                            loss.backward()
                        optimizer.step()
                        torch.cuda.synchronize()

                else:
                    if not sc_flag:
                        outs1, outs2 = dp_model(rank, fc_feats, att_feats, densecap, seq_labels, att_masks, personality)
                        loss1, loss2 = criterion(outs1, seq_labels[:, 1:], seq_masks[:, 1:], outs2, target2)
                        loss = mu*loss1 + (1-mu)*loss2

                    else:
                        gen_result, sample_logprobs, _ = dp_model.module._sample(rank, fc_feats, att_feats, densecap, 
                                                                                 personality=personality, opt={'sample_method': "sample"})
                        scores = get_self_critical_reward(rank, dp_model, fc_feats, att_feats, densecap, personality, gts, gen_result, opt)
                        batch_avg = normalize_baseline(scores, opt.batch_size)
                        running_avg = (i * running_avg + batch_avg) / (i + 1)
                        r_scores = scores[:opt.batch_size] - beta*running_avg*scores[opt.batch_size:]
                        reward = np.repeat(r_scores[:, np.newaxis], gen_result.shape[1], 1)
                        loss = rl_crit(sample_logprobs, gen_result, Variable(torch.from_numpy(reward).float().to(rank), requires_grad=False))
                        store_running_avg.append(r_scores)

                    optimizer.zero_grad()
                    loss.backward()
                    utils.clip_gradient(optimizer, opt.grad_clip)
                    optimizer.step()
                    train_loss = loss.item()

                if i % print_steps == 0:
                    if not sc_flag:
                        print("iter {} (epoch {}), train_loss = {:.3f}".format(i, epoch, train_loss))
                    else:
                        mean_reward = np.mean(reward)
                        print("iter {} (epoch {}), avg_reward = {:.3f}".format(i, epoch, mean_reward))

            # Write the training loss summary
            if (i % opt.losses_log_every == 0):
                loss_history[i] = train_loss if not sc_flag else mean_reward
                lr_history[i] = opt.current_lr
                # ss_prob_history[i] = model.ss_prob

            # update infos
            infos['iter'] = i
            infos['epoch'] = epoch
            
            with open("data/running_avg_rewards.pkl", "wb") as fp:
                pickle.dump(store_running_avg, fp)

            dp_model.eval()
            if (i % opt.save_checkpoint_every == 0):
                # eval model
                eval_kwargs = {'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                val_loss, predictions, lang_stats = eval_utils.eval_split(rank, dp_model, criterion, val_loader, val_dataset, eval_kwargs)

                val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                # Save model if is improving on validation result
                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = - val_loss

                best_flag = False
                print(f"Current Score : {current_score}")
                print(f"Best val score: {best_val_score}")
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                # Dump miscalleous informations
                infos['best_val_score'] = best_val_score
                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history

                if best_flag:
                    save_checkpoint(opt, model, infos, optimizer, histories, append='best')

    except Exception:
        print('Saving checkpoint on exception ...')
        # save_checkpoint(opt, model, infos, optimizer, histories, append="best")
        print('Saving checkpoint done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)


def run_ddp(demo_fn, config, world_size):
    mp.spawn(demo_fn, args=(world_size, config), nprocs=world_size, join=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    opt = opts.parse_opt()
    world_size = 4  # number of gpus to parallize over
    run_ddp(train, opt, world_size)
