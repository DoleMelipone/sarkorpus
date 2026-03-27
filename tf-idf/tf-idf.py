
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import argparse
import pandas as pd

def get_data(dir):
    all_data = dict()
    for year in os.listdir(dir):
        if "y" not in year:
            path = os.path.join(dir, year, 'data', 'by_year_preprocessed.tsv')
            df = pd.read_csv(path, sep='\t')
            for line in df.itertuples():
                if line[1] == 'actual':
                    key = year + '_' + line[2]
                    if key not in all_data:
                        all_data[key] = ''
                    all_data[key] += ' ' + line[4]
    return all_data

def get_data_2(path):
    res = {}
    with open(path) as f:
        data = json.load(f)
    for v in data.values():
        key = str(v['annee']) + '_' + v['journal_clean']
        if key not in res:
            res[key] = ""
        res[key] += (v['titre']+'. '+v['texte'])
    return res


def calculate_tfidf(data, targets, ngram=(1,2)):
    #r"(?u)\b\w+(?:-\w+)*\b"
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+(?:-\w+)*(?:_\w+(?:-\w+)*)+\b", use_idf=True, ngram_range=ngram)
    by_document = []
    for target in targets:
        by_document.append(data[target])

    tfidf = vectorizer.fit_transform(by_document).toarray()
    return vectorizer, tfidf


def save_tfidf_data(vectorizer, tfidf, targets, target, path):
    idx_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}
    targetnum = targets.index(target)
    word_tfidf = dict()

    for idx, value in enumerate(tfidf[targetnum]):
        word = idx_to_word[idx]
        word_tfidf[word] = value

    # sort it from highest to lowest tfidf
    sorted_word_tfidf = sorted(word_tfidf.items(), key=lambda k:k[1], reverse=True)
    # only keep non-zero terms
    sorted_word_tfidf_nonzero = [(k,v) for k,v in sorted_word_tfidf if v > 0]

    if not os.path.exists(path):
        os.mkdir(path)
    [an, source] = target.split('_')
    dir_path = os.path.join(path, an)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    out_path = os.path.join(dir_path, source + "_tfidf.tsv")

    with open(out_path, 'w') as out:
        for w, t in sorted_word_tfidf_nonzero:
            word = ''
            tag = ''
            for sw in w.split():
                [word0, tag0] = sw.split('_')
                word += word0 + ' '
                tag += tag0 + ' '
            word = word.strip()
            tag = tag.strip()
            tag = tag.upper().replace(' ', '+')

            #uppercased_tag = "".join(w.split("_")[:-1]) + "_" + w.split("_")[-1].upper()
            #tag = uppercased_tag.split('_')[-1]
            if 'NOUN' in tag or 'PROPN' in tag:
                wf = word + '_' + tag
                out.write(wf + "\t" + str(t) + "\n")


if __name__ == "__main__":    
    path = os.path.join('datasets', 'sarko_affaires.json')
    dir = 'par_an'
    out_dir = os.path.join('tf-idf', 'tf-idf_par_an_&_source_1&2')

    data = get_data(dir)
    targets = list(data.keys())

    vectorizer, tfidf = calculate_tfidf(data, targets)

    for target in targets:
        save_tfidf_data(vectorizer, tfidf, targets, target, path=out_dir)