import torch.nn.functional as F
import torch.nn as nn
import torch


def topk(output, target, ks=(1,)):
    """Returns one boolean vector for each k, whether the target is within the output's top-k."""
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].max(0)[0] for k in ks]


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if name in skip_list:
                print(name)
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.}]


def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names


def refine_imagenet(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = str(
            list(class_name)).replace(
            '[',
            '').replace(
            ']',
            '').replace(
                '\'',
            '')
    return class_names


def refine_imagenet2(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = str(
            list(class_name)).replace(
            '[',
            '').replace(
            ']',
            '').replace(
                '\'',
                '').replace(
                    ',',
            ', a')
    return class_names


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


index_combine_L14 = [581,
                     1,
                     415,
                     55,
                     337,
                     651,
                     309,
                     300,
                     671,
                     411,
                     496,
                     103,
                     40,
                     654,
                     67,
                     354,
                     628,
                     483,
                     564,
                     345,
                     559,
                     367,
                     892,
                     127,
                     78,
                     831,
                     120,
                     788,
                     572,
                     51,
                     147,
                     385,
                     5,
                     386,
                     277,
                     48,
                     333,
                     935,
                     104,
                     508,
                     846,
                     621,
                     137,
                     291,
                     38,
                     122,
                     269,
                     912,
                     665,
                     970,
                     674,
                     947,
                     574,
                     950,
                     37,
                     357,
                     22,
                     874,
                     717,
                     6,
                     351,
                     729,
                     584,
                     334,
                     298,
                     331,
                     371,
                     149,
                     475,
                     744,
                     119,
                     978,
                     150,
                     2,
                     356,
                     361,
                     967,
                     113,
                     111,
                     73,
                     335,
                     829,
                     985,
                     945,
                     532,
                     847,
                     590,
                     851,
                     292,
                     856,
                     705,
                     0,
                     995,
                     33,
                     922,
                     148,
                     975,
                     272,
                     827,
                     996]

index_rn50 = [948,
              1,
              415,
              358,
              337,
              496,
              309,
              78,
              671,
              440,
              103,
              126,
              540,
              874,
              106,
              354,
              778,
              483,
              64,
              345,
              559,
              367,
              409,
              564,
              314,
              831,
              120,
              357,
              572,
              51,
              148,
              385,
              34,
              31,
              277,
              678,
              333,
              660,
              104,
              508,
              846,
              621,
              384,
              291,
              38,
              122,
              371,
              574,
              665,
              980,
              674,
              947,
              522,
              950,
              584,
              360,
              294,
              392,
              717,
              294,
              812,
              633,
              32,
              334,
              55,
              331,
              380,
              5,
              475,
              744,
              43,
              978,
              150,
              149,
              356,
              361,
              581,
              113,
              66,
              73,
              335,
              829,
              985,
              945,
              532,
              847,
              902,
              851,
              292,
              856,
              705,
              389,
              995,
              33,
              493,
              147,
              912,
              269,
              996,
              111]

argmax_index = [948,
                1,
                680,
                294,
                337,
                831,
                309,
                300,
                671,
                440,
                666,
                416,
                821,
                654,
                323,
                354,
                585,
                483,
                111,
                345,
                559,
                367,
                892,
                812,
                78,
                831,
                120,
                103,
                572,
                51,
                147,
                385,
                5,
                671,
                277,
                416,
                333,
                975,
                104,
                508,
                846,
                621,
                288,
                291,
                38,
                122,
                906,
                912,
                665,
                980,
                674,
                947,
                574,
                950,
                723,
                357,
                978,
                952,
                717,
                975,
                351,
                729,
                985,
                334,
                335,
                330,
                383,
                5,
                475,
                744,
                584,
                978,
                150,
                2,
                335,
                361,
                812,
                113,
                111,
                73,
                335,
                829,
                985,
                945,
                532,
                847,
                528,
                851,
                292,
                866,
                705,
                0,
                995,
                33,
                493,
                147,
                975,
                269,
                459,
                111]
