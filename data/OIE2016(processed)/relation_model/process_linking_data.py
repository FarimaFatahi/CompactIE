import json
import os
from collections import defaultdict
import random

from transformers import AutoTokenizer


def add_marker_tokens(tokenizer, ner_labels):
    new_tokens = ['<START>', '<END>']
    for label in ner_labels:
        new_tokens.append('<START=%s>'%label)
        new_tokens.append('<END=%s>'%label)
    tokenizer.add_tokens(new_tokens)
    print('# vocab after adding markers: %d'%len(tokenizer))


def tokenize_sentences(ext, tokenizer, special_tokens, rel_file):
    rel_indices = {}
    arg_indices = {}
    label_ids = []

    def get_special_token(w):
        if w not in special_tokens:
            special_tokens[w] = ('<' + w + '>').lower()
        return special_tokens[w]

    cls = tokenizer.cls_token
    sep = tokenizer.sep_token
    wordpiece_tokens = [cls]
    wordpiece_tokens_index = []
    cur_index = len(wordpiece_tokens)

    Argument_START_NER = get_special_token("START=Argument")
    Argument_END_NER = get_special_token("END=Argument")
    Relation_START_NER = get_special_token("START=Relation")
    Relation_END_NER = get_special_token("END=Relation")

    ent2offset = {}
    for ent in ext['entityMentions']:
        ent2offset[ent['emId']] = ent['span_ids']

    argument_start_ids = []
    argument_end_ids = []
    relation_start_ids = []
    # add negative relations as well (label = 0)
    relation_end_ids = []
    entity_set = set()
    relation2entity = defaultdict(set)
    for rel in ext['relationMentions']:
        relation_span = ent2offset[rel['arg1']['emId']]
        relation_start_ids.append(relation_span[0])
        relation_end_ids.append(relation_span[-1])
        argument_span = ent2offset[rel['arg2']['emId']]
        argument_start_ids.append(argument_span[0])
        argument_end_ids.append(argument_span[-1])
        label_ids.append(rel_file["id"][rel['label']])
        # add negative sampling
        relation2entity[relation_start_ids[-1]].add(argument_start_ids[-1])
        entity_set.add(argument_start_ids[-1])

    for i, token in enumerate(ext['sentence'].split(' ')):
        if i in relation_start_ids:
            rel_indices[i] = len(wordpiece_tokens)
            wordpiece_tokens.append(Relation_START_NER)
            wordpiece_tokens_index.append([cur_index, cur_index + 1])
            cur_index += 1
        if i in argument_start_ids:
            arg_indices[i] = len(wordpiece_tokens)
            wordpiece_tokens.append(Argument_START_NER)
            wordpiece_tokens_index.append([cur_index, cur_index + 1])
            cur_index += 1

        tokenized_token = list(tokenizer.tokenize(token))
        wordpiece_tokens.extend(tokenized_token)
        wordpiece_tokens_index.append([cur_index, cur_index + len(tokenized_token)])
        cur_index += len(tokenized_token)

        if i in relation_end_ids:
            wordpiece_tokens.append(Relation_END_NER)
            wordpiece_tokens_index.append([cur_index, cur_index + 1])
            cur_index += 1
        if i in argument_end_ids:
            wordpiece_tokens.append(Argument_END_NER)
            wordpiece_tokens_index.append([cur_index, cur_index + 1])
            cur_index += 1

    wordpiece_tokens.append(sep)
    wordpiece_segment_ids = [1] * (len(wordpiece_tokens))
    assert len(argument_start_ids) == len(relation_start_ids)
    assert len(argument_start_ids) == len(label_ids)

    # add negative relations with label 0
    for rel, args in relation2entity.items():
        negative_args = list(entity_set.difference(args))
        for i in range(len(negative_args) // 3):
            arg_index = random.randint(0, len(negative_args) - 1)
            relation_start_ids.append(rel)
            argument_start_ids.append(negative_args[arg_index])
            label_ids.append(0)

    return {
        'sentId': ext['sentId'],
        'sentText': ext['sentence'],
        'entityMentions': ext['entityMentions'],
        'relationMentions': ext['relationMentions'],
        'extractionMentions': ext['extractionMentions'],
        'labelIds': label_ids,
        'relationIds': [rel_indices[r] for r in relation_start_ids],
        'argumentIds': [arg_indices[a] for a in argument_start_ids],
        'wordpieceSentText': " ".join(wordpiece_tokens),
        'wordpieceTokensIndex': wordpiece_tokens_index,
        'wordpieceSegmentIds': wordpiece_segment_ids
    }


def write_dataset_to_file(dataset, dataset_path):
    print("dataset: {}, size: {}".format(dataset_path, len(dataset)))
    with open(dataset_path, 'w', encoding='utf-8') as fout:
        for idx, ext in enumerate(dataset):
            fout.write(json.dumps(ext))
            if idx != len(dataset) - 1:
                fout.write('\n')


def process(source_file, rel_file, target_file, pretrained_model):
    extractions_list = []
    auto_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    print("Load {} tokenizer successfully.".format(pretrained_model))

    rel_id_file = json.load(open(rel_file, 'r', encoding='utf-8'))
    add_marker_tokens(auto_tokenizer, rel_id_file["entity_text"])

    if os.path.exists('special_tokens.json'):
        with open('special_tokens.json', 'r') as f:
            special_tokens = json.load(f)
    else:
        raise FileNotFoundError

    with open(source_file, 'r', encoding='utf-8') as fin, open(target_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            ext = json.loads(line.strip())
            ext_dict = tokenize_sentences(ext, auto_tokenizer, special_tokens, rel_id_file)
            extractions_list.append(ext_dict)
            fout.write(json.dumps(ext_dict))
            fout.write('\n')

    # shuffle and split to train/test/dev
    random.seed(100)
    random.shuffle(extractions_list)
    train_set = extractions_list[:len(extractions_list) - 700]
    dev_set = extractions_list[-700:-200]
    test_set = extractions_list[-200:]
    write_dataset_to_file(train_set, "train.json")
    write_dataset_to_file(dev_set, "devs.json")
    write_dataset_to_file(test_set, "test.json")


if __name__ == '__main__':
    process("../benchmark.json", "rel_file.json", "relation_model_data.json", "bert-base-uncased")