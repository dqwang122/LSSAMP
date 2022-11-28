import os

import numpy as np
import torch
from tensorboardX import SummaryWriter

import pickle as pkl
import json

import distributed
from models.reporter import ReportMgr, Statistics
from others.logging import logger
from models.tokenizer import Vocabulary
# from others.utils import test_rouge, rouge_results_to_str


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

def _tally_trainable_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    return n_params


def build_trainer(args, device_id, model, optims, loss, valid_loss=None):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    device = "cpu" if args.visible_gpus == '-1' else "cuda"


    grad_accum_count = args.accum_count
    n_gpu = args.gpu_num

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)


    trainer = Trainer(args, model, optims, loss, grad_accum_count, n_gpu, gpu_rank, report_manager, valid_loss=valid_loss)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        n_trainable_params = _tally_trainable_parameters(model)
        logger.info('* number of parameters: %d' % n_params)
        logger.info('* number of trainable parameters: %d' % n_trainable_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self,  args, model,  optims, loss,
                  grad_accum_count=1, n_gpu=1, gpu_rank=1,
                  report_manager=None, valid_loss=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optims = optims
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.loss = loss
        self.valid_loss = valid_loss if valid_loss else loss

        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

        self.vocab = Vocabulary(self.args.vocab_path, self.args.vocab_size)

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info('Start training...')

        # step =  self.optim._step + 1
        step =  self.optims[0]._step + 1
        epoch = 0
        epoch_size = 0

        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        epoch_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        best_valid_metric = None
        patience = 0
        earlystopping = False

        while step <= train_steps and not earlystopping:
            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                epoch_size += len(batch)
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    true_batchs.append(batch)
                    num_tokens = batch.tgt[:, 1:].ne(self.loss.padding_idx).sum()
                    normalization += num_tokens.item()
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats)

                        if self.args.save_index:
                            epoch_stats.update(report_stats)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optims[0].learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0

                        if (self.args.train_and_valid and step % valid_steps == 0):
                            valid_stats = self.validate(valid_iter_fct(), step=step)
                            if not best_valid_metric:
                                # best_valid_metric = valid_stats.accuracy()
                                best_valid_metric = valid_stats.xent()
                                self._save_by_name("best")
                            else:
                                # if best_valid_metric < valid_stats.accuracy():
                                #     best_valid_metric = valid_stats.accuracy()
                                if best_valid_metric > valid_stats.xent():
                                    best_valid_metric = valid_stats.xent()
                                    self._save_by_name("best")
                                    patience = 0
                                else:
                                    patience += 1
                            if patience >= self.args.patience:
                                self._save_by_name("last")
                                earlystopping = True
                                # logger.info("Early stopping! The best accuracy is %f", best_valid_metric)
                                logger.info("Early stopping! The best loss is %f", best_valid_metric)
                                break

                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break
            logger.info("Epoch {} finish with {} examples!".format(epoch, epoch_size))
            epoch_size = 0
            train_iter = train_iter_fct()
            if self.args.save_index:
                self._save_index(epoch, epoch_stats)
            epoch_stats = Statistics()
            epoch += 1

        return total_stats

    def validate(self, valid_iter, step=0, is_test=False):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                src = batch.src
                tgt = batch.tgt
                mask_src = batch.mask_src
                mask_tgt = batch.mask_tgt

                if is_test:
                    outputs, _ = self.model.generate(src, tgt, mask_src, mask_tgt)
                else:
                    if self.args.arch == 'cvqvae2' or self.args.arch == 'rnnvqvae2':
                        ss = batch.ss
                        outputs, _ = self.model(src, tgt, ss, mask_src, mask_tgt)
                    else:
                        outputs, _ = self.model(src, tgt, mask_src, mask_tgt)

                batch_stats = self.valid_loss.monolithic_compute_loss(batch, outputs)
                stats.update(batch_stats)
                
            self._report_step(0, step, valid_stats=stats)
            self._save_sample(step, valid_stats=stats)
        self.model.train()
        return stats


    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            tgt = batch.tgt
            mask_src = batch.mask_src
            mask_tgt = batch.mask_tgt

            if self.args.arch == 'cvqvae2' or self.args.arch == 'rnnvqvae2':
                ss = batch.ss
                outputs, index = self.model(src, tgt, ss, mask_src, mask_tgt)
            else:
                outputs, index = self.model(src, tgt, mask_src, mask_tgt)
            batch_stats = self.loss.sharded_compute_loss(batch, outputs, self.args.generator_shard_size, normalization)

            batch_stats.n_docs = int(src.size(0))
            if self.args.save_index and index is not None:
                batch_stats.extra['index'] = index

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))

                for o in self.optims:
                    o.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            for o in self.optims:
                o.step()


    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optims': self.optims,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _save_by_name(self, name):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optims': self.optims,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_%s.pt' % name)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)


    def _save_sample(self, step, valid_stats):
        if valid_stats.extra:
            log_root = self.args.log_dir
            log_dir = log_root + '/{}'.format(self.args.mode)
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)

            with open(os.path.join(log_dir, 'step_{}.log'.format(step)), 'w') as fout:
                string_dict = {}
                cnt = 0
                for k, v in valid_stats.extra.items():
                    if isinstance(v, list):
                        if k.startswith('ss_'):
                            results = ["".join(str(x)) for x in v]
                        else:
                            results = ["".join(self.vocab.ids_to_tokens(x)) for x in v]
                            results = [x.replace('<PAD>', '') for x in results]
                            results = [x.split('</s>')[0] for x in results]
                        string_dict[k] = results
                        
                        cnt = len(string_dict[k])

                for i in range(cnt):
                    for k, v in string_dict.items():
                        fout.write("[{}]\t{}\n".format(k.upper(), v[i]))
                    fout.write('\n')

    def _save_index(self, epoch, train_stats):
        if train_stats.extra and 'index' in train_stats.extra.keys():
            index = train_stats.extra['index']
        else:
            index = []
            return
        
        log_root = self.args.log_dir
        log_dir = log_root + '/{}'.format(self.args.mode)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        save_file = os.path.join(log_dir, 'epoch_{}_index.pt'.format(epoch))
        dataset = []
        for ins in index:
            ex = {}
            ex = {i: x for i, x in enumerate(zip(*ins))}
            dataset.append(ex)
        torch.save(dataset, save_file)



            
            
                            
                            

                

