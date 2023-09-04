import numpy as np
import torch
import torch.nn as nn

from segm.model.blocks import Attention, Block

eps = 1e-8
sigma = 2
mode = 0
layers = [12, 24]

def normalize(score):
    v = np.abs(score)
    v = v / (np.sum(v) + eps)
    return v


def gauss_norm(score, center_score, center):
    if score is []:
        assert 0
    # return score / (score + eps)
    # gauss_filter = [np.exp((-(i-center)**2) / (2*sigma**2)) if score[i] != 0 else 0 for i in range(len(score))]
    # print('gauss_filter: ', gauss_filter)

    # score = np.abs(score * gauss_filter)
    # print("score_before:", score)
    score = score / (center_score + eps)
    bias = score.sum()
    # print('score: ', score)
    return score, bias


def set1_except0(score, center_score, center):
    # print('score:', score)
    value = np.zeros(score.shape[0])
    for i in range(len(score)):
        if score[i] != 0:
            value[i] = 1
    # print('value:', value)
    return value


def normalize_ranks_per_layer(layer_ranks):
    for i in range(len(layer_ranks)):
        v = torch.abs(layer_ranks[i])
        v = v / (torch.sqrt(torch.sum(v * v)) + eps)
        layer_ranks[i] = v
    return layer_ranks


def get_merge_matrix_block(model, scores, center_list, merge_list, recover_list):
    model.cpu()
    stage = 0
    channels = [len(scores[0]) + 1] * layers[mode]
    for name, module in model.named_modules():
        layer_merge = merge_list[stage]
        layer_recover = recover_list[stage]
        score = scores[stage]
        center = center_list[stage]
        square = int(np.sqrt(len(score)))

        # print(mask)

        if isinstance(module, Block) and name == 'blocks.{}'.format(layer_merge):
            print(name)
            # get zero mask
            module.merge = True

            channel = int(len(center)) + 1
            # print('layer {}: {}'.format(layer, channel))
            # channels[layer_merge:layer_recover+1] = channel
            for i in range(layer_merge, layer_recover+1):
                channels[i] = channel

            # init merge matrix
            matrix = np.zeros((channel, len(score) + 1))
            matrix[0][0] = 1
            # print(matrix.shape)

            # init bias
            bias = np.ones(channel)

            left = 1

            for i in range(len(center) - 1):
                line = (center[i] + 1) // square
                right = int(np.ceil((center[i] + center[i + 1]) / 2) + 1)
                # print('before: left:{}, right:{}, next:{}'.format(left, right, center[i + 1] + 1))
                line_r = (right - 1) // square
                if line < line_r:
                    right = (line + 1) * square + 1
                # print('after: left:{}, right:{}, next:{}'.format(left, right, center[i + 1] + 1))
                matrix[i + 1][left:right], bias[i+1] = gauss_norm(score[left - 1:right - 1], score[center[i]], center[i]-left+1)
                # bias[i+1] = (score[left - 1:right - 1] != 0).sum()
                left = right
            matrix[-1][left:], bias[-1] = gauss_norm(score[left - 1:], score[center[-1]], center[-1]-left+1)
            # bias[-1] = (score[left - 1:] != 0).sum()

            # assert 0
            # print(matrix)

            matrix = torch.tensor(matrix, dtype=torch.float32)
            module.merge_matrix = nn.Parameter(matrix)
            # get recover matrix
        if isinstance(module, Block) and name == 'blocks.{}.'.format(layer_recover):
            print(name)
            module.recover = True
            recover_matrix = torch.linalg.pinv(matrix)
            # recover_matrix = torch.zeros((channel, len(score) + 1))
            # for i in range(len(matrix)):
            #     recover_matrix[i] = matrix[i] / (matrix[i].sum() + eps)
            # print(recover_matrix)

            module.recover_matrix = nn.Parameter(recover_matrix)
            # update stage
            if stage < len(merge_list) - 1:
                stage += 1

    return channels

def get_merge_matrix(model, masks, scores, center_list, merge_list):
    model.cpu()
    stage = 0
    channels = [len(scores[0]) + 1] * 12
    for name, module in model.named_modules():
        layer = merge_list[stage]
        mask = masks[stage]
        score = scores[stage]
        center = center_list[stage]
        square = int(np.sqrt(len(score)))

        # print(mask)

        if isinstance(module, Block) and 'blocks.{}'.format(layer) in name:
            # get zero mask
            module.merge = True

            token_mask = torch.cat([torch.tensor([1], dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)], dim=0)
            # print(token_mask)
            module.token_mask = nn.Parameter(token_mask, requires_grad=False)

            channel = int(len(center)) + 1
            # print('layer {}: {}'.format(layer, channel))
            channels[layer] = channel

            # init merge matrix
            matrix = np.zeros((channel, len(score) + 1))
            matrix[0][0] = 1
            # print(matrix.shape)

            # init bias
            bias = np.ones(channel)

            left = 1

            for i in range(len(center) - 1):
                line = (center[i] + 1) // square
                right = int(np.ceil((center[i] + center[i + 1]) / 2) + 1)
                # print('before: left:{}, right:{}, next:{}'.format(left, right, center[i + 1] + 1))
                line_r = (right - 1) // square
                if line < line_r:
                    right = (line + 1) * square + 1
                # print('after: left:{}, right:{}, next:{}'.format(left, right, center[i + 1] + 1))
                matrix[i + 1][left:right], bias[i+1] = gauss_norm(score[left - 1:right - 1], score[center[i]], center[i]-left+1)
                # bias[i+1] = (score[left - 1:right - 1] != 0).sum()
                left = right
            matrix[-1][left:], bias[-1] = gauss_norm(score[left - 1:], score[center[-1]], center[-1]-left+1)
            # bias[-1] = (score[left - 1:] != 0).sum()

            # assert 0
            # print(matrix)

            matrix = torch.tensor(matrix, dtype=torch.float32)
            module.merge_matrix = nn.Parameter(matrix)
            # get recover matrix
            recover_matrix = torch.linalg.pinv(matrix)
            # recover_matrix = torch.zeros((channel, len(score) + 1))
            # for i in range(len(matrix)):
            #     recover_matrix[i] = matrix[i] / (matrix[i].sum() + eps)
            # print(recover_matrix)

            module.recover_matrix = nn.Parameter(recover_matrix)
            # update stage
            if stage < len(merge_list) - 1:
                stage += 1

    return channels


def tm_prune(model, prune_rate, keep_rate, merge_list):
    seq_ranks = []

    for name, module in model.named_modules():
        if isinstance(module, Attention):
            seq_ranks.append(module.seq_ranks.cpu())

    normalize_ranks_per_layer(seq_ranks)
    # print(layer_ranks)`q
    layer_ranks = []
    for i in merge_list:
        layer_ranks.append(np.asarray(seq_ranks[i]) * (1-0.0*i))

    layer_ranks = np.asarray(layer_ranks)
    num_patches = layer_ranks.shape[-1]
    # print(num_patches)

    # calculate for prune and keep numbers
    num_prune = int(prune_rate * num_patches)
    num_keep = int(keep_rate * num_patches)
    # print(int(keep_rate * num_patches))
    # sort the global thr
    # prune_thr = np.sort(np.hstack(layer_ranks))[num_prune]
    # keep_thr = np.sort(np.hstack(layer_ranks))[-num_keep]
    # masks = [layer_rank >= prune_thr for layer_rank in layer_ranks]

    prune_thr = np.sort(layer_ranks, axis=-1)[:, num_prune]
    keep_thr = np.sort(layer_ranks, axis=-1)[:, -num_keep]
    # print(smallest)
    masks = [layer_rank >= thr for layer_rank, thr in zip(layer_ranks, prune_thr)]

    scores = []
    for layer_rank, mask in zip(layer_ranks, masks):
        scores.append(layer_rank * mask)
    # print(scores)
    center_list = [(np.where(score >= thr)[0]) for score, thr in zip(scores, keep_thr)]
    # center_list = [(np.where(score >= keep_thr)[0]) for score in scores]

    # print('center list: ', center_list)
    channels = get_merge_matrix(model, masks, scores, center_list, merge_list)

    print(channels)

    return channels


def tm_prune_block(model, keep_rate, merge_list, recover_list):
    seq_ranks = []

    for name, module in model.named_modules():
        if isinstance(module, Attention):
            seq_ranks.append(module.seq_ranks.cpu())

    normalize_ranks_per_layer(seq_ranks)
    # print(layer_ranks)`q
    layer_ranks = []
    # print(recover_list)
    for i, j in zip(merge_list, recover_list):
        layer_rank = torch.zeros(seq_ranks[i].shape)
        for k in range(i, j+1):
            layer_rank += seq_ranks[k]
        layer_rank = layer_rank / (j - i + 1)
        layer_ranks.append(np.asarray(layer_rank))

    layer_ranks = np.asarray(layer_ranks)
    num_patches = layer_ranks.shape[-1]
    # print(layer_ranks.shape)

    # calculate for prune and keep numbers
    num_keep = int(keep_rate * num_patches)
    # print(int(keep_rate * num_patches))
    # sort the global thr
    # prune_thr = np.sort(np.hstack(layer_ranks))[num_prune]
    # keep_thr = np.sort(np.hstack(layer_ranks))[-num_keep]
    # masks = [layer_rank >= prune_thr for layer_rank in layer_ranks]

    # prune_thr = np.sort(layer_ranks, axis=-1)[:, num_prune]
    keep_thr = np.sort(layer_ranks, axis=-1)[:, -num_keep]
    # print(smallest)
    # masks = [layer_rank >= thr for layer_rank, thr in zip(layer_ranks, prune_thr)]

    # scores = []
    # for layer_rank, mask in zip(layer_ranks, masks):
    #     scores.append(layer_rank * mask)
    # print(scores)
    center_list = [(np.where(score >= thr)[0]) for score, thr in zip(layer_ranks, keep_thr)]
    # center_list = [(np.where(score >= keep_thr)[0]) for score in scores]

    # print('center list: ', center_list)
    channels = get_merge_matrix_block(model, layer_ranks, center_list, merge_list, recover_list)

    print(channels)

    return channels
