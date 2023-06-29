import torch
from argparse import ArgumentParser


def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    if fmap.size(0) == 1:
        fmap = torch.index_select(fmap, 1, index.flatten())
    else:
        index = index.unsqueeze(len(index.shape)).repeat(1, 1, dim)
        fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap


def topk_score(scores, K=8):
    """
    get top K point in score map
    """
    batch, channel, height, width = scores.shape
    topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
    topk_clses = (index / K).int()
    topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
    topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
    topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode(fmap, reg, wh, K=8):
    batch, channel, height, width = fmap.shape
    scores, index, clses, ys, xs = topk_score(fmap, K=K)
    reg = gather_feature(reg, index, use_transform=True)
    reg = reg.reshape(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    wh = gather_feature(wh, index, use_transform=True)
    wh = wh.reshape(batch, K, 2)
    clses = clses.reshape(batch, K, 1).float()
    scores = scores.reshape(batch, K, 1)
    half_w, half_h = wh[..., 0:1] / 2, wh[..., 1:2] / 2
    bboxes = torch.cat([xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim=2)
    detections = (bboxes, scores, clses)
    return detections


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument("-m", "--model", default='rm_centernet_r18d4c6', type=str)
    args.add_argument("-d", "--device", default="CPU", type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=8, type=int)
    args.add_argument("-t", "--threshold", help="threshold for detection", default=0.3, type=float)
    args.add_argument("-p", "--port", help="Port for zeromq. Deploy only", default=5556, type=int)
    args.add_argument("-c", "--color", help="Color for interested armor. Deploy only", default="red", type=str)

    return parser