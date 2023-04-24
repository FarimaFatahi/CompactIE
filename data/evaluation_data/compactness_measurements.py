import pandas as pd
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')
articles = ['a', 'an', 'the']

# input file is the compactIE output extractions on a set of sentences. set this variable accordingly.
INPUT_FILE = 'compactIE_predictions.txt'


def verb_count(part, sent):
    s_doc = nlp(sent)
    text2token = {}
    for i in range(len(s_doc.sentences)):
        tokens = [word.to_dict() for word in s_doc.sentences[i].words]
        for t in tokens:
            text2token[t["text"]] = t
    doc = nlp(part)
    doc = doc.sentences[0]
    tokens = [word.to_dict() for word in doc.words]
    verbs = 0
    for token in tokens:
        if (token['upos'] == 'VERB' and (token['deprel'] not in ['xcomp', 'amod', 'case', 'obl'])) or (token['upos'] == "AUX" and token['deprel'] == 'cop'):
            try:
                if text2token[token["text"]]["deprel"] == token['deprel']:
                    # print(token["text"], token["deprel"])
                    verbs += 1
            except:
                continue
    return verbs


def clausal_constituents(extraction):
    clausal_consts = 0
    if extraction["predicate"].strip() != "":
        pred_count = verb_count(extraction["predicate"], extraction["sentence"])
        if pred_count > 1:
            clausal_consts += pred_count - 1

    if extraction["subject"].strip() != "":
        clausal_consts += verb_count(extraction["subject"], extraction["sentence"])

    if extraction["object"].strip() != "":
        clausal_consts += verb_count(extraction["object"], extraction["sentence"])
    # if clausal_consts > 0:
    #     print("clausal consts within extraction: ", extraction["subject"], extraction["predicate"], extraction["object"], clausal_consts)
    return clausal_consts


if __name__ == "__main__":
    extractions_df = pd.DataFrame(columns=["sentence", "subject", "predicate", "object"])

    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()

    sentences = set()
    for line in lines:
        sentence, ext, score = line.split('\t')
        sentences.add(sentence)
        try:
            arg1 = ext[ext.index('<arg1>') + 6:ext.index('</arg1>')]
        except:
            arg1 = ""
        try:
            rel = ext[ext.index('<rel>') + 5:ext.index('</rel>')]
        except:
            rel = ""
        try:
            arg2 = ext[ext.index('<arg2>') + 6:ext.index('</arg2>')]
        except:
            arg2 = ""
        row = pd.DataFrame(
            {"sentence": [sentence],
             "subject": [arg1],
             "predicate": [rel],
             "object": [arg2]}
        )
        extractions_df = pd.concat([extractions_df, row])

    # overlapping arguments
    grouped_df = extractions_df.groupby("sentence")
    total_number_of_arguments = 0
    number_of_unique_arguments = 0
    num_of_sentences = len(grouped_df.groups.keys())
    for sent in grouped_df.groups.keys():
        per_sentence_argument_set = set()
        sen_group = grouped_df.get_group(sent).reset_index(drop=True)
        extractions_list = list(sen_group.T.to_dict().values())
        for extr in extractions_list:
            if extr["subject"] not in ['', 'nan']:
                total_number_of_arguments += 1
                per_sentence_argument_set.add(extr["subject"])
            if extr["object"] not in ['', 'nan']:
                total_number_of_arguments += 1
                per_sentence_argument_set.add(extr["object"])
        number_of_unique_arguments += len(per_sentence_argument_set)

    print("average # repetitions per argument: {}".format(total_number_of_arguments/number_of_unique_arguments))
    print("average # extractions per sentence: {}".format(extractions_df.shape[0]/len(sentences)))
    avg_arguments_size = 0.0
    for sent in sentences:
        extractions_per_sent = extractions_df[extractions_df["sentence"] == sent]

        sent_extractions = extractions_per_sent.shape[0]
        extractions_per_sent["avg_arg_length"] = extractions_per_sent.apply(lambda r: (len(str(r["subject"]).split(' ')) + len(str(r["predicate"]).split(' ')) + len(str(r["object"]).split(' ')))/3, axis=1)
        avg_arguments_size += sum(extractions_per_sent["avg_arg_length"].values.tolist()) / extractions_per_sent.shape[0]

    print("average length of constituents (per sentence, per extraction): ", avg_arguments_size/len(sentences))
    extractions_df["clause_counts"] = extractions_df.apply(lambda r: clausal_constituents(r), axis=1)
    avg_clause_count = sum(extractions_df["clause_counts"].values.tolist()) / len(sentences)
    print("number of verbal clauses within arguments: ", avg_clause_count)
