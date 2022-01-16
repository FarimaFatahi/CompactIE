from collections import defaultdict
import json
import os
import random
import logging
import time

import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer

from models.joint_decoding.joint_decoder import EntRelJointDecoder
from models.relation_decoding.relation_decoder import RelDecoder
from utils.argparse import ConfigurationParer
from utils.prediction_outputs import print_extractions_allennlp_format
from inputs.vocabulary import Vocabulary
from inputs.fields.token_field import TokenField
from inputs.fields.raw_token_field import RawTokenField
from inputs.fields.map_token_field import MapTokenField
from inputs.instance import Instance
from inputs.datasets.dataset import Dataset
from inputs.dataset_readers.oie_reader_for_ent_rel_decoding import OIE4ReaderForEntRelDecoding

logger = logging.getLogger(__name__)


def step(cfg, ent_model, rel_model, batch_inputs, main_vocab, device):
    batch_inputs["tokens"] = torch.LongTensor(batch_inputs["tokens"])
    batch_inputs["entity_label_matrix"] = torch.LongTensor(batch_inputs["entity_label_matrix"])
    batch_inputs["entity_label_matrix_mask"] = torch.BoolTensor(batch_inputs["entity_label_matrix_mask"])
    batch_inputs["relation_label_matrix"] = torch.LongTensor(batch_inputs["relation_label_matrix"])
    batch_inputs["relation_label_matrix_mask"] = torch.BoolTensor(batch_inputs["relation_label_matrix_mask"])
    batch_inputs["wordpiece_tokens"] = torch.LongTensor(batch_inputs["wordpiece_tokens"])
    batch_inputs["wordpiece_tokens_index"] = torch.LongTensor(batch_inputs["wordpiece_tokens_index"])
    batch_inputs["wordpiece_segment_ids"] = torch.LongTensor(batch_inputs["wordpiece_segment_ids"])

    batch_inputs["joint_label_matrix"] = torch.LongTensor(batch_inputs["joint_label_matrix"])
    batch_inputs["joint_label_matrix_mask"] = torch.BoolTensor(batch_inputs["joint_label_matrix_mask"])

    if device > -1:
        batch_inputs["tokens"] = batch_inputs["tokens"].cuda(device=device, non_blocking=True)
        batch_inputs["entity_label_matrix"] = batch_inputs["entity_label_matrix"].cuda(device=device, non_blocking=True)
        batch_inputs["entity_label_matrix_mask"] = batch_inputs["entity_label_matrix_mask"].cuda(device=device, non_blocking=True)
        batch_inputs["relation_label_matrix"] = batch_inputs["relation_label_matrix"].cuda(device=device, non_blocking=True)
        batch_inputs["relation_label_matrix_mask"] = batch_inputs["relation_label_matrix_mask"].cuda(device=device, non_blocking=True)
        batch_inputs["wordpiece_tokens"] = batch_inputs["wordpiece_tokens"].cuda(device=device, non_blocking=True)
        batch_inputs["wordpiece_tokens_index"] = batch_inputs["wordpiece_tokens_index"].cuda(device=device, non_blocking=True)
        batch_inputs["wordpiece_segment_ids"] = batch_inputs["wordpiece_segment_ids"].cuda(device=device, non_blocking=True)

    ent_outputs = ent_model(batch_inputs, rel_model, main_vocab)
    batch_outputs = []
    if not ent_model.training and not rel_model.training:
        # entities
        for sent_idx in range(len(batch_inputs['tokens_lens'])):
            sent_output = dict()
            sent_output['tokens'] = batch_inputs['tokens'][sent_idx].cpu().numpy()
            sent_output['span2ent'] = batch_inputs['span2ent'][sent_idx]
            sent_output['span2rel'] = batch_inputs['span2rel'][sent_idx]
            sent_output['seq_len'] = batch_inputs['tokens_lens'][sent_idx]
            sent_output['entity_label_matrix'] = batch_inputs['entity_label_matrix'][sent_idx].cpu().numpy()
            sent_output['entity_label_preds'] = ent_outputs['entity_label_preds'][sent_idx].cpu().numpy()
            sent_output['separate_positions'] = batch_inputs['separate_positions'][sent_idx]
            sent_output['all_separate_position_preds'] = ent_outputs['all_separate_position_preds'][sent_idx]
            sent_output['all_ent_preds'] = ent_outputs['all_ent_preds'][sent_idx]
            sent_output['all_rel_preds'] = ent_outputs['all_rel_preds'][sent_idx]
            batch_outputs.append(sent_output)
        return batch_outputs

    return ent_outputs['element_loss'], ent_outputs['symmetric_loss']


def test(cfg, dataset, ent_model, rel_model):
    logger.info("Testing starting...")
    ent_model.zero_grad()
    rel_model.zero_grad()

    all_outputs = []
    cost_time = 0
    idx = 0
    # with tqdm(dataset, unit="batch") as tepoch:
    for _, batch in dataset.get_batch('test', cfg.test_batch_size, None):
        print("{} processed".format(idx+1))
        idx += 1
        ent_model.eval()
        rel_model.eval()
        with torch.no_grad():
            cost_time -= time.time()
            batch_outputs = step(cfg, ent_model, rel_model, batch, dataset.vocab, cfg.device)
            cost_time += time.time()
        all_outputs.extend(batch_outputs)

    logger.info(f"Cost time: {cost_time}s")
    logger.info(f"Cost time: {cost_time}s")
    test_output_file = os.path.join(cfg.save_dir, "output_extractions.txt")
    print_extractions_allennlp_format(cfg, all_outputs, test_output_file, dataset.vocab)
    # print_predictions_for_entity_rel_decoding(all_outputs, test_output_file, dataset.vocab)


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
    separate_positions = RawTokenField("separate_positions", "separate_positions")
    span2ent = MapTokenField("span2ent", "ent_rel_id", "span2ent", False)
    span2rel = MapTokenField("span2rel", "ent_rel_id", "span2rel", False)
    entity_label_matrix = RawTokenField("entity_label_matrix", "entity_label_matrix")
    relation_label_matrix = RawTokenField("relation_label_matrix", "relation_label_matrix")
    joint_label_matrix = RawTokenField("joint_label_matrix", "joint_label_matrix")
    wordpiece_tokens = TokenField("wordpiece_tokens", "wordpiece", "wordpiece_tokens", False)
    wordpiece_tokens_index = RawTokenField("wordpiece_tokens_index", "wordpiece_tokens_index")
    wordpiece_segment_ids = RawTokenField("wordpiece_segment_ids", "wordpiece_segment_ids")
    fields = [tokens, separate_positions, span2ent, span2rel, entity_label_matrix, relation_label_matrix, joint_label_matrix]

    if cfg.embedding_model in ['bert', 'pretrained']:
        fields.extend([wordpiece_tokens, wordpiece_tokens_index, wordpiece_segment_ids])

    # define counter and vocabulary
    counter = defaultdict(lambda: defaultdict(int))
    vocab_ent = Vocabulary()

    # define instance (data sets)
    test_instance = Instance(fields)

    # define dataset reader
    max_len = {'tokens': cfg.max_sent_len, 'wordpiece_tokens': cfg.max_wordpiece_len}
    ent_rel_file = json.load(open(cfg.ent_rel_file, 'r', encoding='utf-8'))
    rel_file = json.load(open(cfg.rel_file, 'r', encoding='utf-8'))
    pretrained_vocab = {'ent_rel_id': ent_rel_file["id"]}
    if cfg.embedding_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained(cfg.bert_model_name)
        logger.info("Load bert tokenizer successfully.")
        pretrained_vocab['wordpiece'] = tokenizer.get_vocab()
    elif cfg.embedding_model == 'pretrained':
        tokenizer = BertTokenizer.from_pretrained(cfg.pretrained_model_name)
        logger.info("Load {} tokenizer successfully.".format(cfg.pretrained_model_name))
        pretrained_vocab['wordpiece'] = tokenizer.get_vocab()
    oie_test_reader = OIE4ReaderForEntRelDecoding(cfg.test_file, False, max_len)

    # define dataset
    oie_dataset = Dataset("OIE4")
    oie_dataset.add_instance("test", test_instance, oie_test_reader, is_count=True, is_train=False)

    min_count = {"tokens": 1}
    no_pad_namespace = ["ent_rel_id"]
    no_unk_namespace = ["ent_rel_id"]
    contain_pad_namespace = {"wordpiece": tokenizer.pad_token}
    contain_unk_namespace = {"wordpiece": tokenizer.unk_token}
    oie_dataset.build_dataset(vocab=vocab_ent,
                              counter=counter,
                              min_count=min_count,
                              pretrained_vocab=pretrained_vocab,
                              no_pad_namespace=no_pad_namespace,
                              no_unk_namespace=no_unk_namespace,
                              contain_pad_namespace=contain_pad_namespace,
                              contain_unk_namespace=contain_unk_namespace)
    wo_padding_namespace = ["separate_positions", "span2ent", "span2rel"]
    oie_dataset.set_wo_padding_namespace(wo_padding_namespace=wo_padding_namespace)

    vocab_ent = Vocabulary.load(cfg.constituent_vocab)
    vocab_rel = Vocabulary.load(cfg.relation_vocab)
    print("ent vocab: ", cfg.constituent_vocab)
    # separate models for constituent generation and linking
    ent_model = EntRelJointDecoder(cfg=cfg, vocab=vocab_ent, ent_rel_file=ent_rel_file, rel_file=rel_file)
    rel_model = RelDecoder(cfg=cfg, vocab=vocab_rel, ent_rel_file=rel_file)

    # main bert-based model
    if os.path.exists(cfg.constituent_model_path):
        print(cfg.constituent_model_path)
        state_dict = torch.load(open(cfg.constituent_model_path, 'rb'), map_location=lambda storage, loc: storage)
        ent_model.load_state_dict(state_dict)
        print("constituent model loaded")
    if os.path.exists(cfg.relation_model_path):
        state_dict = torch.load(open(cfg.relation_model_path, 'rb'), map_location=lambda storage, loc: storage)
        rel_model.load_state_dict(state_dict)
        print("linking model loaded")
    else:
        raise FileNotFoundError
    logger.info("Loading best training models successfully for testing.")

    if cfg.device > -1:
        ent_model.cuda(device=cfg.device)
        rel_model.cuda(device=cfg.device)

    test(cfg, oie_dataset, ent_model, rel_model)


if __name__ == '__main__':
    main()
