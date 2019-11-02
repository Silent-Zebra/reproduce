# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
from fairseq import utils

from . import FairseqCriterion, register_criterion


def out_disagreement(attn_variables):
    # Retrieve context (shape bsz x nheads x L x dhead), mask (shape bsz x L) and weights (shape bsz x nheads x L x l)
    attn = attn_variables["attn"]
    out_mask = attn_variables["out_mask"]

    # Reverse mask
    # if out_mask is not None:
    #     out_mask = torch.eq(out_mask, 0.0).float()
    # else:
    out_mask = torch.ones(attn.size(0), attn.size(2)).to(attn.device)

    # avg output disagreement
    out = attn / (torch.sqrt((attn ** 2).sum(-1, keepdim=True)) + 1e-7)
    out_dis = torch.einsum("blid,bljd->blij", [out, out])
    # Reduce
    out_dis = (out_dis * out_mask.unsqueeze(-1).unsqueeze(-1)).mean()
    return out_dis


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disagreement-reg', default=0., type=float, metavar='D',
                            help='Disagreement regularization')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        # Add disagreement regularization
        if getattr(self.args, "disagreement_reg", 0.0) > 0:
            alpha = getattr(self.args, "disagreement_reg", 0.0)
            encoder_layers = self.args.encoder_layers
            decoder_layers = self.args.decoder_layers
            encoder_heads = self.args.encoder_attention_heads
            decoder_heads = self.args.decoder_attention_heads
            reg_loss = 0
            for layer in range(encoder_layers):
                self_attn_variables = model.encoder.layers[layer].self_attn_variables
                reg_loss += out_disagreement(self_attn_variables)
            # Retrieve importance scores for the decoder
            for layer in range(decoder_layers):
                # Self attention
                self_attn_variables = model.decoder.layers[layer].self_attn_variables
                reg_loss += out_disagreement(self_attn_variables)
                encoder_attn_variables = model.decoder.layers[layer].encoder_attn_variables
                reg_loss += out_disagreement(encoder_attn_variables)
        loss += alpha * reg_loss

        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
