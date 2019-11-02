# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from fairseq import utils

from . import FairseqDataset


def translate_samples(samples, collate_fn, generate_fn, cuda=True):
    """translate a list of samples.

    Given an input (*samples*) of the form:

        [{'id': 1, 'source': 'hallo welt'}]

    this will return:

        [{'id': 1, 'source': 'hello world', 'target': 'hallo welt'}]

    Args:
        samples (List[dict]): samples to translate. Individual samples are
            expected to have a 'source' key, which will become the 'target'
            after translation.
        collate_fn (callable): function to collate samples into a mini-batch
        generate_fn (callable): function to generate translations
        cuda (bool): use GPU for generation (default: ``True``)

    Returns:
        List[dict]: an updated list of samples with a translated source
    """
    collated_samples = collate_fn(samples)
    s = utils.move_to_cuda(collated_samples) if cuda else collated_samples
    generated_targets = generate_fn(s['net_input'])

    def update_sample(sample, generated_target):
        # the original source becomes the target
        sample['target'] = generated_target
        return sample

    # Go through each src sentence in batch and its corresponding best
    # generated hypothesis and create a translation data pair
    # {id: id, source: generated translation, target: original src}
    return [
        update_sample(
            sample=input_sample,
            # highest scoring hypo is first
            generated_target=hypos[0]['tokens'].cpu(),
        )
        for input_sample, hypos in zip(samples, generated_targets)
    ]


class TranslationDataset(FairseqDataset):
    def __init__(
        self,
        src_dataset,
        translation_fn,
        max_len_a,
        max_len_b,
        output_collater=None,
        cuda=True,
        **kwargs
    ):
        """
        Sets up a translation dataset which takes a src batch, generates
        a tgt using a src-tgt translation function (*translation_fn*),
        and returns the corresponding `{input src, generated tgt}` batch.

        Args:
            src_dataset (~fairseq.data.FairseqDataset): the dataset to be
                translated. Only the source side of this dataset will be
                used. After translation, the source sentences in this
                dataset will be returned as the targets.
            translation_fn (callable): function to call to generate
                translations. This is typically the `generate` method of a
                :class:`~fairseq.sequence_generator.SequenceGenerator` object.
            max_len_a, max_len_b (int, int): will be used to compute
                `maxlen = max_len_a * tgt_len + max_len_b`, which will be
                passed into *translation_fn*.
            output_collater (callable, optional): function to call on the
                translated samples to create the final batch (default:
                ``src_dataset.collater``)
            cuda: use GPU for generation
        """
        self.src_dataset = src_dataset
        self.translation_fn = translation_fn
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.output_collater = output_collater if output_collater is not None \
            else src_dataset.collater
        self.cuda = cuda if torch.cuda.is_available() else False

    def __getitem__(self, index):
        """
        Returns a single sample from *src_dataset*. Note that translation is
        not applied in this step; use :func:`collater` instead to translate
        a batch of samples.
        """
        return self.src_dataset[index]

    def __len__(self):
        return len(self.src_dataset)

    def collater(self, samples):
        """Merge and translate a list of samples to form a mini-batch.

        Using the samples from *src_dataset*, load a collated source sample to
        feed to the translation model. Then take the translation with
        the best score as the source and the original input as the source.

        Note: we expect *src_dataset* to provide a function `collater()` that
        will collate samples into the format expected by *translation_fn*.
        After translation, we will feed the new list of samples (i.e., the
        `(translated source, original source)` pairs) to *output_collater*
        and return the result.

        Args:
            samples (List[dict]): samples to translate and collate

        Returns:
            dict: a mini-batch with keys coming from *output_collater*
        """
        samples = translate_samples(
            samples=samples,
            collate_fn=self.src_dataset.collater,
            generate_fn=(
                lambda net_input: self.translation_fn(
                    {"src_tokens": net_input["src_tokens"],
                     "src_lengths": net_input["src_lengths"]},
                    maxlen=int(
                        self.max_len_a *
                        net_input['src_tokens'].size(1) + self.max_len_b
                    ),
                )
            ),
            cuda=self.cuda,
        )
        return self.output_collater(samples)

    def get_dummy_batch(self, num_tokens, max_positions):
        """Just use the src dataset get_dummy_batch"""
        return self.src_dataset.get_dummy_batch(num_tokens, max_positions)

    def num_tokens(self, index):
        """Just use the src dataset num_tokens"""
        return self.src_dataset.num_tokens(index)

    def ordered_indices(self):
        """Just use the src dataset ordered_indices"""
        return self.src_dataset.ordered_indices()

    def valid_size(self, index, max_positions):
        """Just use the src dataset size"""
        return self.src_dataset.valid_size(index, max_positions)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used
        when filtering a dataset with ``--max-positions``.

        Note: we use *src_dataset* to approximate the length of the source
        sentence, since we do not know the actual length until after
        translation.
        """
        src_size = self.src_dataset.size(index)[0]
        return (src_size, src_size)

    def prefetch(self, indices):
        self.src_dataset.prefetch(indices)

    @property
    def supports_prefetch(self):
        return getattr(self.src_dataset, 'supports_prefetch', False)

