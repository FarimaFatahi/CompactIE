from collections import defaultdict
import json
import os
import random
import logging
import time

import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from utils.argparse import ConfigurationParer
from inputs.vocabulary import Vocabulary
from inputs.fields.token_field import TokenField
from inputs.fields.raw_token_field import RawTokenField
from inputs.instance import Instance
from inputs.datasets.dataset import Dataset
from inputs.dataset_readers.oie_reader_for_relation_detection import ReaderForRelationDecoding
from models.relation_decoding.relation_decoder import RelDecoder
from utils.nn_utils import get_n_trainable_parameters

logger = logging.getLogger(__name__)


def step(cfg, model, batch_inputs, device):
    batch_inputs["tokens"] = torch.LongTensor(batch_inputs["tokens"])
    batch_inputs["label_ids"] = torch.LongTensor(batch_inputs["label_ids"])
    batch_inputs["label_ids_mask"] = torch.BoolTensor(batch_inputs["relation_ids_mask"])
    batch_inputs["relation_ids"] = torch.LongTensor(batch_inputs["relation_ids"])
    batch_inputs["relation_ids_mask"] = torch.BoolTensor(batch_inputs["relation_ids_mask"])
    batch_inputs["argument_ids"] = torch.LongTensor(batch_inputs["argument_ids"])
    batch_inputs["argument_ids_mask"] = torch.BoolTensor(batch_inputs["argument_ids_mask"])
    batch_inputs["wordpiece_tokens"] = torch.LongTensor(batch_inputs["wordpiece_tokens"])
    batch_inputs["wordpiece_tokens_mask"] = torch.BoolTensor(batch_inputs["wordpiece_tokens_mask"])
    batch_inputs["wordpiece_tokens_index"] = torch.LongTensor(batch_inputs["wordpiece_tokens_index"])
    batch_inputs["wordpiece_segment_ids"] = torch.LongTensor(batch_inputs["wordpiece_segment_ids"])

    if device > -1:
        batch_inputs["tokens"] = batch_inputs["tokens"].cuda(device=device, non_blocking=True)
        batch_inputs["label_ids"] = batch_inputs["label_ids"].cuda(device=device, non_blocking=True)
        batch_inputs["label_ids_mask"] = batch_inputs["label_ids_mask"].cuda(device=device, non_blocking=True)
        batch_inputs["relation_ids"] = batch_inputs["relation_ids"].cuda(device=device,non_blocking=True)
        batch_inputs["relation_ids_mask"] = batch_inputs["relation_ids_mask"].cuda(device=device, non_blocking=True)
        batch_inputs["argument_ids"] = batch_inputs["argument_ids"].cuda(device=device,non_blocking=True)
        batch_inputs["argument_ids_mask"] = batch_inputs["argument_ids_mask"].cuda(device=device, non_blocking=True)
        batch_inputs["wordpiece_tokens"] = batch_inputs["wordpiece_tokens"].cuda(device=device, non_blocking=True)
        batch_inputs["wordpiece_tokens_mask"] = batch_inputs["wordpiece_tokens_mask"].cuda(device=device, non_blocking=True)
        batch_inputs["wordpiece_tokens_index"] = batch_inputs["wordpiece_tokens_index"].cuda(device=device,
                                                                                             non_blocking=True)
        batch_inputs["wordpiece_segment_ids"] = batch_inputs["wordpiece_segment_ids"].cuda(device=device,
                                                                                           non_blocking=True)

    outputs = model(batch_inputs)
    batch_outputs = []
    if not model.training:
        for sent_idx in range(len(batch_inputs['tokens_lens'])):
            sent_output = dict()
            sent_output['tokens'] = batch_inputs['tokens'][sent_idx].cpu().numpy()
            sent_output['label_ids'] = batch_inputs['label_ids'][sent_idx].cpu().numpy()
            sent_output['relation_ids'] = batch_inputs['relation_ids'][sent_idx].cpu().numpy()
            sent_output['argument_ids'] = batch_inputs['argument_ids'][sent_idx].cpu().numpy()
            sent_output['seq_len'] = batch_inputs['tokens_lens'][sent_idx]
            sent_output['label_preds'] = outputs['label_preds'][sent_idx].cpu().numpy()
            batch_outputs.append(sent_output)
        return batch_outputs

    return outputs['loss']


def train(cfg, dataset, model):
    logger.info("Training starting...")

    for name, param in model.named_parameters():
        logger.info("{!r}: size: {} requires_grad: {}.".format(name, param.size(), param.requires_grad))

    logger.info("Trainable parameters size: {}.".format(get_n_trainable_parameters(model)))

    parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_layer_lr = {}
    base_lr = cfg.bert_learning_rate
    for i in range(11, -1, -1):
        bert_layer_lr['.' + str(i) + '.'] = base_lr
        base_lr *= cfg.lr_decay_rate

    optimizer_grouped_parameters = []
    for name, param in parameters:
        params = {'params': [param], 'lr': cfg.learning_rate}
        if any(item in name for item in no_decay):
            params['weight_decay_rate'] = 0.0
        else:
            if 'bert' in name:
                params['weight_decay_rate'] = cfg.adam_bert_weight_decay_rate
            else:
                params['weight_decay_rate'] = cfg.adam_weight_decay_rate

        for bert_layer_name, lr in bert_layer_lr.items():
            if bert_layer_name in name:
                params['lr'] = lr
                break

        optimizer_grouped_parameters.append(params)

    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(cfg.adam_beta1, cfg.adam_beta2),
                      lr=cfg.learning_rate,
                      eps=cfg.adam_epsilon,
                      weight_decay=cfg.adam_weight_decay_rate,
                      correct_bias=False)

    total_train_steps = (dataset.get_dataset_size("train") + cfg.train_batch_size * cfg.gradient_accumulation_steps -
                         1) / (cfg.train_batch_size * cfg.gradient_accumulation_steps) * cfg.epochs
    num_warmup_steps = int(cfg.warmup_rate * total_train_steps) + 1
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_train_steps)

    last_epoch = 1
    batch_id = 0
    best_f1 = 0.0
    early_stop_cnt = 0
    accumulation_steps = 0
    model.zero_grad()

    for epoch, batch in dataset.get_batch('train', cfg.train_batch_size, None):

        if last_epoch != epoch or (batch_id != 0 and batch_id % cfg.validate_every == 0):
            if accumulation_steps != 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if epoch > cfg.pretrain_epochs:
                dev_f1 = dev(cfg, dataset, model)
                if dev_f1 > best_f1:
                    early_stop_cnt = 0
                    best_f1 = dev_f1
                    logger.info("Save model...")
                    torch.save(model.state_dict(), open(cfg.best_model_path, "wb"))
                elif last_epoch != epoch:
                    early_stop_cnt += 1
                    if early_stop_cnt > cfg.early_stop:
                        logger.info("Early Stop: best F1 score: {:6.2f}%".format(100 * best_f1))
                        break
        if epoch > cfg.epochs:
            torch.save(model.state_dict(), open(cfg.last_model_path, "wb"))
            logger.info("Training Stop: best F1 score: {:6.2f}%".format(100 * best_f1))
            break

        if last_epoch != epoch:
            batch_id = 0
            last_epoch = epoch

        model.train()
        batch_id += len(batch['tokens_lens'])
        batch['epoch'] = (epoch - 1)
        loss = step(cfg, model, batch, cfg.device)
        if batch_id % cfg.logging_steps == 0:
            logger.info("Epoch: {} Batch: {} Loss: {})".format(epoch, batch_id, loss.item()))

        if cfg.gradient_accumulation_steps > 1:
            loss /= cfg.gradient_accumulation_steps

        loss.backward()

        accumulation_steps = (accumulation_steps + 1) % cfg.gradient_accumulation_steps
        if accumulation_steps == 0:
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=cfg.gradient_clipping)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

    state_dict = torch.load(open(cfg.best_model_path, "rb"), map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    test(cfg, dataset, model)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_f1(output):
    n_gold = n_pred = n_correct = 0
    for sent in output:
        for pred, label in zip(sent["label_preds"], sent["label_ids"]):
            if pred != 0:
                n_pred += 1
            if label != 0:
                n_gold += 1
            if (pred != 0) and (label != 0) and (pred == label):
                n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0

        return {'precision': prec, 'task_recall': recall, 'task_f1': f1,
        'n_correct': n_correct, 'n_pred': n_pred, 'task_ngold': n_gold}


def evaluate(outputs):
    result = compute_f1(outputs)
    logger.info("Validation F1: {}, Accuracy: {}, Recall: {}".format(result["task_f1"], result["precision"], result["task_recall"]))
    return result["task_f1"]


def dev(cfg, dataset, model):
    logger.info("Validate starting...")
    model.zero_grad()

    all_outputs = []
    cost_time = 0
    for _, batch in dataset.get_batch('dev', cfg.test_batch_size, None):
        model.eval()
        with torch.no_grad():
            cost_time -= time.time()
            batch_outpus = step(cfg, model, batch, cfg.device)
            cost_time += time.time()
        all_outputs.extend(batch_outpus)
    logger.info(f"Cost time: {cost_time}s")
    f1 = evaluate(all_outputs)
    return f1


def test(cfg, dataset, model):
    logger.info("Testing starting...")
    model.zero_grad()

    all_outputs = []

    cost_time = 0
    for _, batch in dataset.get_batch('test', cfg.test_batch_size, None):
        model.eval()
        with torch.no_grad():
            cost_time -= time.time()
            batch_outpus = step(cfg, model, batch, cfg.device)
            cost_time += time.time()
        all_outputs.extend(batch_outpus)
    logger.info(f"Cost time: {cost_time}s")

    f1 = evaluate(all_outputs)
    print("test F1: ", f1)


def main():
    # config settings
    parser = ConfigurationParer()
    parser.add_save_cfgs()
    parser.add_data_cfgs()
    parser.add_model_cfgs()
    parser.add_optimizer_cfgs()
    parser.add_run_cfgs()

    cfg = parser.parse_args()
    logger.info(parser.format_values())

    # set random seed
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.device > -1 and not torch.cuda.is_available():
        logger.error('config conflicts: no gpu available, use cpu for training.')
        cfg.device = -1
    if cfg.device > -1:
        torch.cuda.manual_seed(cfg.seed)

    # define fields
    tokens = TokenField("tokens", "tokens", "tokens", True)
    label_ids = RawTokenField("label_ids", "label_ids")
    relation_ids = RawTokenField("relation_ids", "relation_ids")
    argument_ids = RawTokenField("argument_ids", "argument_ids")
    wordpiece_tokens = TokenField("wordpiece_tokens", "wordpiece", "wordpiece_tokens", False)
    wordpiece_tokens_index = RawTokenField("wordpiece_tokens_index", "wordpiece_tokens_index")
    wordpiece_segment_ids = RawTokenField("wordpiece_segment_ids", "wordpiece_segment_ids")
    fields = [tokens, label_ids, relation_ids, argument_ids]

    if cfg.embedding_model in ['bert', 'pretrained']:
        fields.extend([wordpiece_tokens, wordpiece_tokens_index, wordpiece_segment_ids])

    # define counter and vocabulary
    counter = defaultdict(lambda: defaultdict(int))
    vocab = Vocabulary()

    # define instance (data sets)
    train_instance = Instance(fields)
    dev_instance = Instance(fields)
    test_instance = Instance(fields)

    # define dataset reader
    max_len = {'tokens': cfg.max_sent_len, 'wordpiece_tokens': cfg.max_wordpiece_len}
    ent_rel_file = json.load(open(cfg.ent_rel_file, 'r', encoding='utf-8'))
    pretrained_vocab = {'ent_rel_id': ent_rel_file["id"]}
    if cfg.embedding_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained(cfg.bert_model_name)
        logger.info("Load bert tokenizer successfully.")
        pretrained_vocab['wordpiece'] = tokenizer.get_vocab()
    elif cfg.embedding_model == 'pretrained':
        tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name)
        logger.info("Load {} tokenizer successfully.".format(cfg.pretrained_model_name))
        pretrained_vocab['wordpiece'] = tokenizer.get_vocab()
    train_reader = ReaderForRelationDecoding(cfg.train_file, False, max_len)
    dev_reader = ReaderForRelationDecoding(cfg.dev_file, False, max_len)
    test_reader = ReaderForRelationDecoding(cfg.test_file, False, max_len)

    # define dataset
    oie_dataset = Dataset("OIE4")
    oie_dataset.add_instance("train", train_instance, train_reader, is_count=True, is_train=True)
    oie_dataset.add_instance("dev", dev_instance, dev_reader, is_count=True, is_train=False)
    oie_dataset.add_instance("test", test_instance, test_reader, is_count=True, is_train=False)

    min_count = {"tokens": 1}
    no_pad_namespace = ["ent_rel_id"]
    no_unk_namespace = ["ent_rel_id"]
    contain_pad_namespace = {"wordpiece": tokenizer.pad_token}
    contain_unk_namespace = {"wordpiece": tokenizer.unk_token}
    oie_dataset.build_dataset(vocab=vocab,
                              counter=counter,
                              min_count=min_count,
                              pretrained_vocab=pretrained_vocab,
                              no_pad_namespace=no_pad_namespace,
                              no_unk_namespace=no_unk_namespace,
                              contain_pad_namespace=contain_pad_namespace,
                              contain_unk_namespace=contain_unk_namespace)
    oie_dataset.set_wo_padding_namespace(wo_padding_namespace=[])
    if cfg.test:
        vocab = Vocabulary.load(cfg.vocabulary_file)
    else:
        vocab.save(cfg.vocabulary_file)

    # rel model
    model = RelDecoder(cfg=cfg, vocab=vocab, ent_rel_file=ent_rel_file)

    if cfg.test and os.path.exists(cfg.best_model_path):
        state_dict = torch.load(open(cfg.best_model_path, 'rb'), map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        logger.info("Loading best training model {} successfully for testing.".format(cfg.best_model_path))

    if cfg.device > -1:
        model.cuda(device=cfg.device)

    if cfg.test:
        dev(cfg, oie_dataset, model)
        test(cfg, oie_dataset, model)
    else:
        train(cfg, oie_dataset, model)


if __name__ == '__main__':
    main()
