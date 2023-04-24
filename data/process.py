import json
import argparse
import sys
from collections import defaultdict
from transformers import AutoTokenizer


def read_conjunctive_sentences(args):
    with open(args.conjunctions_file, 'r') as fin:
        sent = True
        sent2conj = defaultdict(list)
        conj2sent = dict()
        currentSentText = ''
        for line in fin:
            if line == '\n':
                sent = True
                continue
            if sent:
                currentSentText = line.replace('\n', '')
                sent = False
            else:
                conj_sent = line.replace('\n', '')
                sent2conj[currentSentText].append(conj_sent)
                conj2sent[conj_sent] = currentSentText

    return sent2conj


def get_conj_free_sentence_dicts(sentence, sent_to_conj, sent_id):
    flat_extractions_list = []
    sentence = sentence.replace('\n', '')
    if sentence in list(sent_to_conj.keys()):
        for s in sent_to_conj[sentence]:
            sentence_and_extractions_dict = {
                "sentence": s + " [unused1] [unused2] [unused3] [unused4] [unused5] [unused6]",
                "sentId": sent_id, "entityMentions": list(),
                "relationMentions": list(), "extractionMentions": list()}
            flat_extractions_list.append(sentence_and_extractions_dict)
        return flat_extractions_list

    return [{
        "sentence": sentence + " [unused1] [unused2] [unused3] [unused4] [unused5] [unused6]",
        "sentId": sent_id, "entityMentions": list(),
        "relationMentions": list(), "extractionMentions": list()}]


def add_joint_label(ext, ent_rel_id):
    """add_joint_label add joint labels for sentences
    """

    none_id = ent_rel_id['None']
    sentence_length = len(ext['sentText'].split(' '))
    entity_label_matrix = [[none_id for j in range(sentence_length)] for i in range(sentence_length)]
    relation_label_matrix = [[none_id for j in range(sentence_length)] for i in range(sentence_length)]
    label_matrix = [[none_id for j in range(sentence_length)] for i in range(sentence_length)]
    ent2offset = {}
    for ent in ext['entityMentions']:
        ent2offset[ent['emId']] = ent['span_ids']
        try:
            for i in ent['span_ids']:
                for j in ent['span_ids']:
                    entity_label_matrix[i][j] = ent_rel_id[ent['label']]
        except:
            print("span ids: ", sentence_length, ent['span_ids'], ext)
            sys.exit(1)
    ext['entityLabelMatrix'] = entity_label_matrix
    for rel in ext['relationMentions']:
        arg1_span = ent2offset[rel['arg1']['emId']]
        arg2_span = ent2offset[rel['arg2']['emId']]

        for i in arg1_span:
            for j in arg2_span:
                # to be consistent with the linking model
                relation_label_matrix[i][j] = ent_rel_id[rel['label']] - 2
                relation_label_matrix[j][i] = ent_rel_id[rel['label']] - 2
                label_matrix[i][j] = ent_rel_id[rel['label']]
                label_matrix[j][i] = ent_rel_id[rel['label']]
    ext['relationLabelMatrix'] = relation_label_matrix
    ext['jointLabelMatrix'] = label_matrix


def tokenize_sentences(ext, tokenizer):
    cls = tokenizer.cls_token
    sep = tokenizer.sep_token
    wordpiece_tokens = [cls]

    wordpiece_tokens_index = []
    cur_index = len(wordpiece_tokens)
    # for token in ext['sentText'].split(' '):
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


def process(args, sent2conj):
    extractions_list = []
    auto_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    print("Load {} tokenizer successfully.".format(args.embedding_model))

    ent_rel_id = json.load(open(args.ent_rel_file, 'r', encoding='utf-8'))["id"]
    sentId = 0
    with open(args.source_file, 'r', encoding='utf-8') as fin, open(args.target_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            sentId += 1
            exts = get_conj_free_sentence_dicts(line, sent2conj, sentId)
            for ext in exts:
                # ext = ext.strip()
                ext_dict = tokenize_sentences(ext, auto_tokenizer)
                add_joint_label(ext_dict, ent_rel_id)
                extractions_list.append(ext_dict)
                fout.write(json.dumps(ext_dict))
                fout.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process sentences.')
    parser.add_argument("--source_file", type=str, help='source file path')
    parser.add_argument("--target_file", type=str, help='target file path')
    parser.add_argument("--conjunctions_file", type=str, help='conjunctions file.')
    parser.add_argument("--ent_rel_file", type=str, default="ent_rel_file.json", help='entity and relation file.')
    parser.add_argument("--embedding_model", type=str, default="bert-base-uncased", help='embedding model.')

    args = parser.parse_args()
    sent2conj = read_conjunctive_sentences(args)
    process(args, sent2conj)
