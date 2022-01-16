import json
import random
import sys
from transformers import AutoTokenizer


def add_joint_label(ext, ent_rel_id):
    """add_joint_label add joint labels for sentences
    """

    none_id = ent_rel_id['None']
    sentence_length = len(ext['sentText'].split(' '))
    label_matrix = [[none_id for j in range(sentence_length)] for i in range(sentence_length)]
    ent2offset = {}
    for ent in ext['entityMentions']:
        ent2offset[ent['emId']] = ent['span_ids']
        try:
            for i in ent['span_ids']:
                for j in ent['span_ids']:
                    label_matrix[i][j] = ent_rel_id[ent['label']]
        except:
            sys.exit(1)
    for rel in ext['relationMentions']:
        arg1_span = ent2offset[rel['arg1']['emId']]
        arg2_span = ent2offset[rel['arg2']['emId']]

        for i in arg1_span:
            for j in arg2_span:
                # symmetric relations
                label_matrix[i][j] = ent_rel_id[rel['label']]
                label_matrix[j][i] = ent_rel_id[rel['label']]
    ext['jointLabelMatrix'] = label_matrix


def tokenize_sentences(ext, tokenizer):
    cls = tokenizer.cls_token
    sep = tokenizer.sep_token
    wordpiece_tokens = [cls]

    wordpiece_tokens_index = []
    cur_index = len(wordpiece_tokens)
    for token in ext['sentence'].split(' '):
        tokenized_token = list(tokenizer.tokenize(token))
        wordpiece_tokens.extend(tokenized_token)
        wordpiece_tokens_index.append([cur_index, cur_index + len(tokenized_token)])
        cur_index += len(tokenized_token)
    wordpiece_tokens.append(sep)

    wordpiece_segment_ids = [1] * (len(wordpiece_tokens))

    return {
        'sentId': ext['sentId'],
        'sentText': ext['sentence'],
        'entityMentions': ext['entityMentions'],
        'relationMentions': ext['relationMentions'],
        'extractionMentions': ext['extractionMentions'],
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


def process(source_file, ent_rel_file, target_file, pretrained_model, max_length=50):
    extractions_list = []
    auto_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    print("Load {} tokenizer successfully.".format(pretrained_model))

    ent_rel_id = json.load(open(ent_rel_file, 'r', encoding='utf-8'))["id"]

    with open(source_file, 'r', encoding='utf-8') as fin, open(target_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            ext = json.loads(line.strip())
            ext_dict = tokenize_sentences(ext, auto_tokenizer)
            add_joint_label(ext_dict, ent_rel_id)
            extractions_list.append(ext_dict)
            fout.write(json.dumps(ext_dict))
            fout.write('\n')

    # shuffle and split to train/test/dev
    random.shuffle(extractions_list)
    train_set = extractions_list[:len(extractions_list) - 700]
    dev_set = extractions_list[-700:-200]
    test_set = extractions_list[-200:]
    write_dataset_to_file(train_set, "joint_model_data_albert/train.json")
    write_dataset_to_file(dev_set, "joint_model_data_albert/dev.json")
    write_dataset_to_file(test_set, "joint_model_data_albert/test.json")


if __name__ == '__main__':
    process("../benchmark.json", "ent_rel_file.json", "constituent_model_data.json", "bert-base-uncased")