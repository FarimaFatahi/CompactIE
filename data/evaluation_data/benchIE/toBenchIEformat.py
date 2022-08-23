sent2id = dict()
with open('sample300_en.txt', 'r') as f:
    i = 1
    for line in f.readlines():
        sent2id[line.replace('\n', '')] = i
        i += 1

with open('output_extractions.txt', 'r') as fin, open('compactIE_1.2_new.txt', 'w') as fout:
    for line in fin.readlines():
        sentence, extraction, score = line.split('\t')
        sentId = sent2id[sentence]
        try:
            arg1 = extraction[extraction.index('<arg1>') + 6:extraction.index('</arg1>')]
            arg1 = arg1.strip()
        except:
            print("subject error!", extraction)
            arg1 = ""
        try:
            rel = extraction[extraction.index('<rel>') + 5:extraction.index('</rel>')]
            rel = rel.strip()
        except:
            print("predicate error!", extraction)
            rel = ""
        try:
            arg2 = extraction[extraction.index('<arg2>') + 6:extraction.index('</arg2>')]
            arg2 = arg2.strip()
            if arg2 == "":
                continue
        except:
            print("object error!", extraction)
            arg2 = ""
        print("{}\t{}\t{}\t{}".format(sentId, arg1, rel, arg2), file=fout)