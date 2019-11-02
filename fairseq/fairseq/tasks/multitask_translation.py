import itertools
import os

from fairseq import options
from fairseq.sequence_generator import SequenceGenerator
from fairseq.data import (
    data_utils, Dictionary, LanguagePairDataset, TranslationDataset,
    IndexedRawTextDataset, IndexedCachedDataset, IndexedDataset
)

from . import FairseqTask, register_task


@register_task('multitask_translation')
class MultiTaskTranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>` and :mod:`interactive.py <interactive>`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', nargs='+',
                            help='path(s) to data directorie(s)')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--translate-primary', action='store_true',
                            help='Translate from the primary dataset')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        print("Setup multitask")
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                'Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = Dictionary.load(os.path.join(
            args.data[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = Dictionary.load(os.path.join(
            args.data[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(
            args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(
            args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        print("Load multitask data")

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(
                data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        src_datasets = []
        tgt_datasets = []

        data_paths = self.args.data

        for dk, data_path in enumerate(data_paths):
            print(f"loading {data_path}")
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(
                        data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(
                        data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError(
                            'Dataset not found: {} ({})'.format(split, data_path))

                src_datasets.append(indexed_dataset(
                    prefix + src, self.src_dict))
                tgt_datasets.append(indexed_dataset(
                    prefix + tgt, self.tgt_dict))

                print('| {} {} {} examples'.format(
                    data_path, split_k, len(src_datasets[-1])))

                if not combine:
                    break

        assert len(src_datasets) == len(tgt_datasets)

        if self.args.translate_primary and split is "train":
            src_dataset = LanguagePairDataset(
                src_datasets[0], src_datasets[0].sizes, self.src_dict,
                tgt_datasets[0], tgt_datasets[0].sizes, self.tgt_dict,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )
            self.datasets[split] = [
                TranslationDataset(
                    src_dataset=src_dataset,
                    # We will set the translation_fn later when the model is loaded
                    translation_fn=None,
                    max_len_a=0,
                    max_len_b=200,
                    # output_collater=TransformEosDataset(
                    #    src_dataset,
                    #    eos=self.tgt_dict.eos(),
                    #    append_eos_to_tgt=True,
                    # ).collater,
                    # cuda=True,
                ),
                LanguagePairDataset(
                    src_datasets[1], src_datasets[1].sizes, self.src_dict,
                    tgt_datasets[1], tgt_datasets[1].sizes, self.tgt_dict,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                )
            ]
        else:
            self.datasets[split] = [
                LanguagePairDataset(
                    src_dataset, src_dataset.sizes, self.src_dict,
                    tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                )
                for src_dataset, tgt_dataset
                in zip(src_datasets, tgt_datasets)
            ]

    def dataset(self, split, idx=0):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """
        from fairseq.data import FairseqDataset
        if split not in self.datasets:
            raise KeyError('Dataset not loaded: ' + split)
        if not isinstance(self.datasets[split][idx], FairseqDataset):
            raise TypeError(
                'Datasets are expected to be of type FairseqDataset')
        if idx < 0 or idx >= len(self.args.data):
            raise ValueError(f'Invalid dataset idx: {idx}')
        return self.datasets[split][idx]

    def build_model(self, args):
        model = super(MultiTaskTranslationTask, self).build_model(args)
        if self.args.translate_primary:
            self.generator = SequenceGenerator(
                models=[model],
                tgt_dict=self.tgt_dict,
                beam_size=1,
                unk_penalty=0,
                sampling=True,
            )
            for split in self.datasets:
                self.datasets[split][0].translation_fn = self.generator.generate
        return model

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
