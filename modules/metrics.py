import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats
from config.all_config import gen_log
import gc

def np_softmax(X, theta = 1.0, axis = None):

    y = np.atleast_2d(X)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1:
        p = p.flatten()
    return p

def sim_matrix_training(text_embeds, vid_embeds_pooled, pooling_type):

    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())
        
    else:
        vid_embeds_pooled = vid_embeds_pooled.permute(1, 2, 0)
        text_embeds = text_embeds.unsqueeze(1)
        
        sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)

    return sims


def generate_embeds_per_video_id(text_embeds, vid_embeds_pooled, all_vid_ids, pooling_type):
    text_embeds_per_video_id = {}

    for idx, v_id in enumerate(all_vid_ids):
        if v_id in text_embeds_per_video_id:
            text_embeds_per_video_id[v_id].append(text_embeds[idx])
        else:
            text_embeds_per_video_id[v_id] = [text_embeds[idx]]

    for v_id in text_embeds_per_video_id:
        text_embeds_per_video_id[v_id] = torch.stack(text_embeds_per_video_id[v_id])

    text_embeds_per_video_id = pad_and_stack_dict_to_tensor(text_embeds_per_video_id,
        text_embeds_per_video_id.keys(), text_embeds.shape[-1])

    if pooling_type == 'avg':
        vid_embeds_pooled_per_video_id = vid_embeds_pooled

    else:
        vid_embeds_pooled_per_video_id = []

        for i in range(vid_embeds_pooled.shape[0]):
            vid_embeds_pooled_per_video_id.append({})
            for idx, v_id in enumerate(all_vid_ids):
                if v_id in vid_embeds_pooled_per_video_id[i]:
                    vid_embeds_pooled_per_video_id[i][v_id].append(vid_embeds_pooled[i, idx, :])
                else:
                    vid_embeds_pooled_per_video_id[i][v_id] = [vid_embeds_pooled[i, idx, :]]

        for i in range(len(vid_embeds_pooled_per_video_id)):
            for v_id in vid_embeds_pooled_per_video_id[i]:
                vid_embeds_pooled_per_video_id[i][v_id] = torch.stack(vid_embeds_pooled_per_video_id[i][v_id])

            vid_embeds_pooled_per_video_id[i] = pad_and_stack_dict_to_tensor(vid_embeds_pooled_per_video_id[i],
                    vid_embeds_pooled_per_video_id[i].keys(), vid_embeds_pooled.shape[-1])

        vid_embeds_pooled_per_video_id = torch.stack(vid_embeds_pooled_per_video_id)

    return text_embeds_per_video_id, vid_embeds_pooled_per_video_id

def sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, pooling_type):

    text_embeds_per_video_id = text_embeds_per_video_id / text_embeds_per_video_id.norm(dim=-1, keepdim=True)
    vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id / vid_embeds_pooled_per_video_id.norm(dim=-1, keepdim=True)

    if pooling_type == 'avg':
        print(f'for this case, have not tried')
        raise NotImplementedError

    else:
        num_txts, num_vids, max_text_per_vid, embed_dim = text_embeds_per_video_id.shape

        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.permute(1, 2, 3, 0)
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.reshape(num_vids * max_text_per_vid, embed_dim, num_vids)
        text_embeds_per_video_id = text_embeds_per_video_id.permute(0, 2, 1, 3)
        text_embeds_per_video_id = text_embeds_per_video_id.reshape(num_vids * max_text_per_vid, num_txts, embed_dim)

        sims = torch.bmm(text_embeds_per_video_id, vid_embeds_pooled_per_video_id)
        sims = sims.view(num_vids, max_text_per_vid, num_txts, num_vids)
        sims_diag = torch.stack([sims[i, :, :, i] for i in range(sims.shape[0])], dim=-1)
        print(f'>>>check sims_diag={sims_diag.shape}')
        sims_diag = sims_diag.permute(1, 0, 2)

    return sims_diag


def sim_matrix_inference_light_allops(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, pooling_type, batch_size_split, config):

    text_embeds_per_video_id = text_embeds_per_video_id / text_embeds_per_video_id.norm(dim=-1, keepdim=True)
    vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id / vid_embeds_pooled_per_video_id.norm(dim=-1, keepdim=True)
    if pooling_type == 'avg':
        print(f'for this case, have not tried')
        raise NotImplementedError
    else:
        vid_embeds_pooled_per_video_id = vid_embeds_pooled_per_video_id.permute(1, 2, 3, 0)
        text_embeds_per_video_id = text_embeds_per_video_id.permute(0, 2, 1, 3)

        batch_size = text_embeds_per_video_id.shape[0]
        if batch_size_split is None:
            batch_size_split = 1
        else:
            pass
        dim0, dim1, dim2, dim3 = text_embeds_per_video_id.shape
        sims_diag = torch.zeros(dim1, dim0, dim2)

        for batch in range(0, batch_size, batch_size_split):
            tensor1_batch = text_embeds_per_video_id[batch: min(batch + batch_size_split, batch_size)]
            tensor2_batch = vid_embeds_pooled_per_video_id[batch: min(batch + batch_size_split, batch_size)]

            result_batch = torch.matmul(tensor1_batch, tensor2_batch)

            for idx in range(batch, min(batch + batch_size_split, batch_size)):
                sims_diag[:, :, idx] = result_batch[idx - batch, :, :, idx]
        del text_embeds_per_video_id, vid_embeds_pooled_per_video_id
        gc.collect()

        print(f'>>>check sims_diag={sims_diag.shape}')

        sims_diag = sims_diag.permute(1, 0, 2)

    return sims_diag

def generate_embeds_per_video_id(text_embeds_stochastic_allpairs, vid_embeds_pooled, all_vid_ids, pooling_type):

    if pooling_type == 'avg':
        text_embeds_per_video_id = text_embeds_stochastic_allpairs

    else:
        text_embeds_per_video_id = []

        for i in range(text_embeds_stochastic_allpairs.shape[0]):
            text_embeds_per_video_id.append({})
            for idx, t_id in enumerate(all_vid_ids):
                if t_id in text_embeds_per_video_id[i]:
                    text_embeds_per_video_id[i][t_id].append(text_embeds_stochastic_allpairs[i, idx, :])
                else:
                    text_embeds_per_video_id[i][t_id] = [text_embeds_stochastic_allpairs[i, idx, :]]

        for i in range(len(text_embeds_per_video_id)):
            for t_id in text_embeds_per_video_id[i]:
                text_embeds_per_video_id[i][t_id] = torch.stack(text_embeds_per_video_id[i][t_id])

            text_embeds_per_video_id[i] = pad_and_stack_dict_to_tensor(text_embeds_per_video_id[i],
                                                                       text_embeds_per_video_id[i].keys(),
                                                                       text_embeds_stochastic_allpairs.shape[-1])

        text_embeds_per_video_id = torch.stack(text_embeds_per_video_id)

    if pooling_type == 'avg':
        vid_embeds_pooled_per_video_id = vid_embeds_pooled

    else:
        vid_embeds_pooled_per_video_id = []

        for i in range(vid_embeds_pooled.shape[0]):
            vid_embeds_pooled_per_video_id.append({})
            for idx, v_id in enumerate(all_vid_ids):
                if v_id in vid_embeds_pooled_per_video_id[i]:
                    vid_embeds_pooled_per_video_id[i][v_id].append(vid_embeds_pooled[i, idx, :])
                else:
                    vid_embeds_pooled_per_video_id[i][v_id] = [vid_embeds_pooled[i, idx, :]]

        for i in range(len(vid_embeds_pooled_per_video_id)):
            for v_id in vid_embeds_pooled_per_video_id[i]:
                vid_embeds_pooled_per_video_id[i][v_id] = torch.stack(vid_embeds_pooled_per_video_id[i][v_id])

            vid_embeds_pooled_per_video_id[i] = pad_and_stack_dict_to_tensor(vid_embeds_pooled_per_video_id[i],
                                                                             vid_embeds_pooled_per_video_id[i].keys(),
                                                                             vid_embeds_pooled.shape[-1])

        vid_embeds_pooled_per_video_id = torch.stack(vid_embeds_pooled_per_video_id)

    return text_embeds_per_video_id, vid_embeds_pooled_per_video_id

def metrics(x):
    x = x[:, 0, :]
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['R50'] = float(np.sum(ind < 50)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MdR"] = metrics['MR']
    metrics["MnR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics

def pad_and_stack_dict_to_tensor(input, order, d=512):
    max_length = max([input[k].shape[0] for k in input])
    
    padded_input = {k: torch.cat([input[k], torch.full((max_length - input[k].shape[0], d), float("-inf"), device=input[k].device)]) for k in input}
    
    padded_stacked_input = torch.stack([padded_input[k] for k in order], dim=0)
    return padded_stacked_input
