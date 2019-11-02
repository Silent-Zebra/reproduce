#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import numpy as np
import sys
import copy

import torch

from fairseq import data, options, tasks, tokenizer, utils
from fairseq.sequence_generator import SequenceGenerator


Batch = namedtuple('Batch', 'srcs tokens lengths')
Translation = namedtuple(
    'Translation', 'src_str hypos pos_scores alignments')


def buffered_read(buffer_size, input_feed=None):
    if input_feed is None:
        input_feed = sys.stdin
    buffer = []
    for src_str in input_feed:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, task, max_positions, max_sentences, max_tokens):
    tokens = [
        tokenizer.Tokenizer.tokenize(
            src_str, task.source_dictionary, add_if_not_exist=False).long()
        for src_str in lines
    ]
    lengths = np.array([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=data.LanguagePairDataset(
            tokens, lengths, task.source_dictionary),
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            srcs=[lines[i] for i in batch['id']],
            tokens=batch['net_input']['src_tokens'],
            lengths=batch['net_input']['src_lengths'],
        ), batch['id']


def parse_head_pruning_descriptors(
    descriptors,
    reverse_descriptors=False,
    n_heads=None
):
    """Returns a dictionary mapping layers to the set of heads to prune in
    this layer (for each kind of attention)"""
    to_prune = {
        "E": {},
        "A": {},
        "D": {},
    }
    for descriptor in descriptors:
        attn_type, layer, heads = descriptor.split(":")
        layer = int(layer) - 1
        heads = set(int(head) - 1 for head in heads.split(","))
        if layer not in to_prune[attn_type]:
            to_prune[attn_type][layer] = set()
        to_prune[attn_type][layer].update(heads)
    # Reverse
    if reverse_descriptors:
        if n_heads is None:
            raise ValueError("You need to specify the total number of heads")
        for attn_type in to_prune:
            for layer, heads in to_prune[attn_type].items():
                to_prune[attn_type][layer] = set([head for head in range(n_heads)
                                                  if head not in heads])
    return to_prune


def get_attn_layer(model, attn_type, layer):
    if attn_type == "E":
        return model.encoder.layers[layer].self_attn
    elif attn_type == "D":
        return model.decoder.layers[layer].self_attn
    elif attn_type == "A":
        return model.decoder.layers[layer].encoder_attn


def make_result(src_str, hypos, align_dict, tgt_dict, nbest=1, remove_bpe=False, print_alignment=False):
    result = Translation(
        src_str='O\t{}'.format(src_str),
        hypos=[],
        pos_scores=[],
        alignments=[],
    )
    # Process top predictions
    for i, hypo in enumerate(hypos[:min(len(hypos), nbest)]):
        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
            hypo_tokens=hypo['tokens'].int().cpu(),
            src_str=src_str,
            alignment=hypo['alignment'].int().cpu(
            ) if hypo['alignment'] is not None else None,
            align_dict=align_dict,
            tgt_dict=tgt_dict,
            remove_bpe=remove_bpe,
        )
        # Now all hypos
        result.hypos.append('H\t{}\t{}'.format(hypo['score'], hypo_str))
        result.pos_scores.append('P\t{}'.format(
            ' '.join(map(
                lambda x: '{:.4f}'.format(x),
                hypo['positional_scores'].tolist(),
            ))
        ))
        result.alignments.append(
            'A\t{}'.format(
                ' '.join(map(lambda x: str(utils.item(x)), alignment)))
            if print_alignment else None
        )
    return result


def process_batch(
    translator,
    batch,
    align_dict,
    tgt_dict,
    use_cuda=False,
    nbest=1,
    remove_bpe=False,
    print_alignment=False,
    max_len_a=0,
    max_len_b=200,
):
    tokens = batch.tokens
    lengths = batch.lengths

    if use_cuda:
        tokens = tokens.cuda()
        lengths = lengths.cuda()

    encoder_input = {'src_tokens': tokens, 'src_lengths': lengths}
    translations = translator.generate(
        encoder_input,
        maxlen=int(max_len_a * tokens.size(1) + max_len_b),
    )

    batch_results = [
        make_result(
            batch.srcs[i],
            t,
            align_dict,
            tgt_dict,
            nbest=nbest,
            remove_bpe=remove_bpe,
            print_alignment=print_alignment,
        ) for i, t in enumerate(translations)
    ]
    return batch_results


def mask_heads(model, to_prune, rescale=False):
    for attn_type in to_prune:
        for layer, heads in to_prune[attn_type].items():
            attn_layer = get_attn_layer(model, attn_type, layer)
            attn_layer.mask_heads = heads
            attn_layer.mask_head_rescale = rescale
            attn_layer._head_mask = None


def translate_corpus(
    translator,
    task,
    input_feed=None,
    buffer_size=1,
    replace_unk=False,
    use_cuda=False,
    print_directly=False,
    nbest=1,
    remove_bpe=False,
    print_alignment=False,
    max_sentences=1,
    max_tokens=9999,
):
    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in translator.models]
    )
    if input_feed is None:
        input_feed = sys.stdin

    if buffer_size > 1:
        print('| Sentence buffer size:', buffer_size)
    print('| Type the input sentence and press return:')
    all_results = []
    for inputs in buffered_read(buffer_size, input_feed):
        indices = []
        results = []
        for batch, batch_indices in make_batches(inputs, task, max_positions, max_sentences, max_tokens):
            indices.extend(batch_indices)
            results += process_batch(
                translator,
                batch,
                align_dict,
                copy.deepcopy(task.target_dictionary),
                use_cuda=use_cuda,
                nbest=nbest,
                remove_bpe=remove_bpe,
                print_alignment=print_alignment,
            )
        # Sort results
        results = [results[i] for i in np.argsort(indices)]
        # Print to stdout
        if print_directly:
            for result in results:
                print(result.src_str)
                for hypo, pos_scores, align in zip(result.hypos, result.pos_scores, result.alignments):
                    print(hypo)
                    print(pos_scores)
                    if align is not None:
                        print(align)
        all_results.extend(results)
    return all_results


def main(args):
    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model_paths = args.path.split(':')
    models, model_args = utils.load_ensemble_for_inference(
        model_paths,
        task,
        model_arg_overrides=eval(args.model_overrides)
    )

    # Set dictionaries
    tgt_dict = copy.deepcopy(task.target_dictionary)

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()

    if len(args.transformer_mask_heads) > 0:
        # Determine which head to prune
        to_prune = parse_head_pruning_descriptors(
            args.transformer_mask_heads,
            reverse_descriptors=args.transformer_mask_all_but_one_head,
            n_heads=model.encoder.layers[0].self_attn.num_heads
        )
        print(to_prune)
        # Apply pruning
        mask_heads(model, to_prune, args.transformer_mask_rescale)

    # Initialize generator
    translator = SequenceGenerator(
        models, tgt_dict, beam_size=args.beam, minlen=args.min_len,
        stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
        len_penalty=args.lenpen, unk_penalty=args.unkpen,
        sampling=args.sampling, sampling_topk=args.sampling_topk, sampling_temperature=args.sampling_temperature,
        diverse_beam_groups=args.diverse_beam_groups, diverse_beam_strength=args.diverse_beam_strength,
    )

    if use_cuda:
        translator.cuda()

    translate_corpus(
        translator,
        task,
        buffer_size=args.buffer_size,
        replace_unk=args.replace_unk,
        use_cuda=use_cuda,
        print_directly=True,
        nbest=args.nbest,
        remove_bpe=args.remove_bpe,
        print_alignment=args.print_alignment,
        max_sentences=args.max_sentences,
        max_tokens=args.max_tokens,
    )


if __name__ == '__main__':
    parser = options.get_generation_parser(interactive=True)
    options.add_pruning_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)
