from collections import defaultdict, Counter
import pandas as pd
import json
import re
import os
import dill
from transformers import AutoTokenizer
from nltk.collocations import *
from nltk import sent_tokenize
from french_stopwords import french_stopwords
import spacy
from itertools import islice
import numpy as np

#!!!!!!!!!!!!
nlp = spacy.load("fr_core_news_lg")
#!!!!!!!!!!!!

w_tokenizer = AutoTokenizer.from_pretrained("camembert-base", use_fast=False)

names1 = {'Le Figaro': 'figaro', 'Libération': 'liberation'}
names2 = {'Libération': 'liberation', 'Le Monde': 'monde'}
names3 = {'Le Figaro': 'figaro', 'Le Monde': 'monde'}
comps = {'libe-fig': names1, 'libe-monde': names2, 'monde-fig': names3}

political_orientation = {'Le Figaro': 'right', 'Libération': 'left', 'Le Monde': 'center'}

class Vocab():
    def __init__(self):
        self.docs = defaultdict(list)
        self.lemmatized_docs = defaultdict(list)
        self.chunks = []
        self.meta = defaultdict(list)

    def add(self, doc, lemmatized_doc, chunk, source):
        if chunk not in self.chunks:
            self.chunks.append(chunk)
        self.docs[chunk].append(doc)
        self.lemmatized_docs[chunk].append(lemmatized_doc)
        self.meta[chunk].append(source)

    def make_vocab(self, vocab_path):
        print('making_vocab')
        all_freqs = []
        freqs = defaultdict(int)
        punctuation = "!#%'()*+,.:;=?@[\]^`{|}~"
        sw = french_stopwords
        for chunk in self.chunks:
            print("chunk: ", chunk)
            chunk_freqs = defaultdict(int)
            count_words = 0
            for doc in self.lemmatized_docs[chunk]:
                for sent in doc.split(" <eos> "):
                    for word in sent.split():
                        test_word = word.split('_')[0]
                        is_punct = False
                        for p in punctuation:
                            if p in word:
                                is_punct = True
                                break
                        if not is_punct:
                            is_digit = word.isdigit()
                            if not is_digit:
                                if len(test_word) > 2 and test_word.lower() not in sw:
                                    chunk_freqs[word] += 1
                                    freqs[word] += 1
                                    count_words += 1
            all_freqs.append((chunk_freqs, count_words))
        print('All vocab size: ', len(freqs))

        filtered_freqs = []
        biggest_diff = []
        for word, freq in freqs.items():
            allow = True
            diff = []
            for chunk_freq, _ in all_freqs:
                diff.append(chunk_freq[word])
                if chunk_freq[word] < 10:
                    allow = False
                    break
            if len(diff) == 2:
                diff = abs(diff[0] - diff[1])
                biggest_diff.append((word, diff))
            if allow:
                filtered_freqs.append((word, freq))

        print('Len filtered freq: ', len(filtered_freqs))
        self.freqs = []
        freqs = sorted(filtered_freqs, key=lambda x: x[1], reverse=True)
        with open(vocab_path, 'w', encoding='utf8') as f:
            f.write('word,mean\n')
            for w, freq in freqs:
                w = w_tokenizer.tokenize(w)
                #w = "".join(w).replace('##', '')
                w = "".join(w).replace('▁', ' ').strip()
                f.write(w + ',' + str(freq) + '\n')
                self.freqs.append((w, freq))


def remove_url(text, replace_token):
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(regex, replace_token, text)


def preprocess_doc(text):
    text = text.replace('~', '').replace('­', '').replace('▲', '').replace('_', '') \
        .replace('■', '').replace('*', '').replace('^', '').replace('<', '').replace('"', '') \
        .replace('�', '').replace('/', ' ').replace('“', '').replace('”', '').replace('"', '') \
        .replace('-', ' ').replace('–', ' ').replace('—', ' ').replace('–', ' ')
    text = remove_url(text, '')
    text = re.sub("[\(\[].*?[\)\]]", "", text)
    sents = sent_tokenize(text)
    text_filtered = []
    for sent in sents:
        corrupted = False
        if not sent.isupper():
            sent = sent.split()
            sent = " ".join(sent)
            if len(sent) <= 3:
                corrupted = True
            if not corrupted:
                text_filtered.append(sent)

    clean_doc = []
    lemmatized_doc = []
    for sent in text_filtered:
        sent = nlp(sent)
        original_sent = []
        lemmatized_sent = []
        iterator = iter(range(len(sent)))
        for i in iterator:
            token = sent[i]
            lemma = token.lemma_.lower()
            pos = token.pos_
            text_token = token.text
            if '’' in text_token:
                lemma = text_token
            if token.ent_iob_ == 'B':
                j = i
                while j < len(sent)-1:
                    j += 1
                    next_token = sent[j]
                    if next_token.ent_iob_ == 'I':
                        lemma = lemma + '-' + next_token.lemma_.lower()
                        text_token = text_token + '-' + next_token.text
                        pos = 'PROPN'
                        next(islice(iterator, 1, 1), None)
                    else:
                        break
            lemmatized_sent.append(lemma + '_' + pos)
            original_sent.append(text_token)

        original_sent = " ".join(original_sent)
        lemmatized_sent = " ".join(lemmatized_sent)
        clean_doc.append(original_sent)
        lemmatized_doc.append(lemmatized_sent)
    if len(clean_doc) > 0 and len(lemmatized_doc) > 0:
        return clean_doc, lemmatized_doc
    return None


def preprocess_corpus(corpus_path, output_path, year, comp):
    all_data = []
    all_sents = []

    counter = 0
    count_period = defaultdict(int)
    duplicates = set()
    with open(corpus_path) as f:
        data = json.load(f)
    print("Num docs: ", len(data.keys()))
    for idx, row in data.items():
        date = row['annee']
        if row['titre'] not in duplicates and date == year:
            duplicates.add(row['titre'])
            counter += 1
            publisher = row['journal_clean'].strip()

            #political orientation
            po_name = ''
            po = ''
            names = comps[comp]
            if publisher in names:
                po_name = names[publisher]
                po = political_orientation[publisher]
            if po_name:
                count_period[po_name] += 1
                period = str(year)
                text = row['titre'] + '. ' + row['texte']
                output = preprocess_doc(text)
                if output is not None:
                    text, lemmatized_text = output
                    all_sents.extend(text)
                    all_data.append((po_name, po, period, " <eos> ".join(text), " <eos> ".join(lemmatized_text)))
                    #print("Original:", " <eos> ".join(text)[:100])
                    #print("Lemmatized:", " <eos> ".join(lemmatized_text)[:100])
                    #print('------------------------------------------')
                else:
                    print('Discarded doc', text)

    df = pd.DataFrame(all_data, columns=['publisher', 'po', 'period', 'text', 'lemmatized_text'])
    df.to_csv(output_path, sep='\t', encoding='utf8', index=False)
    print('Corpus preprocessed')
    print('Period counts', Counter(count_period))

def filter_artefacts(df):
    num_filtered = 0
    other = []
    sent_freqs = defaultdict(int)
    for idx, row in df.iterrows():
        text = row['text']
        sents = text.split('<eos>')
        for sent in sents:
            sent_freqs[sent] += 1

    for idx, row in df.iterrows():
        text = row['text']
        lemmas = row['lemmatized_text']
        sents = text.split('<eos>')
        lemmas = lemmas.split('<eos>')
        filtered_sents = []
        filtered_lemmas = []
        for i, sent in enumerate(sents):
            if sent_freqs[sent] < 10:
                filtered_sents.append(sent)
                filtered_lemmas.append(lemmas[i])
            else:
                other.append(sent)
                num_filtered += 1
        #print(df.loc[idx, 'lemmatized_text'])
        sents = '<eos>'.join(filtered_sents)
        lemmas = '<eos>'.join(filtered_lemmas)
        df.loc[idx, 'lemmatized_text'] = lemmas
        #print(df.loc[idx, 'lemmatized_text'])
        #print('-------')
        df.loc[idx, 'text'] = sents
    print('Num sents filtered: ', num_filtered)
    #df.to_csv('corpus_filtered.tsv', sep='\t', encoding='utf8', index=False)
    return df



if __name__ == '__main__':
    corpus_path = os.path.join('datasets', 'sarko_affaires.json')

    years = [y for y in range(2015, 2026)]
    for year in years:
        for comp in comps:
            f = os.path.join('by_year_newspapers', comp, str(year))
            if not os.path.exists(f):
                os.mkdir(f)
            folder = os.path.join(f, 'data')
            if not os.path.exists(folder):
                os.mkdir(folder)
            vocab_output = os.path.join(folder, 'by_year_vocab.pickle')
            vocab_path = os.path.join(folder, 'by_year_vocab.csv')
            output_path = os.path.join(folder, 'by_year_preprocessed.tsv')

            preprocess_corpus(corpus_path, output_path, year, comp)
            df_data = pd.read_csv(output_path, sep='\t', encoding='utf8')
            df_data = df_data.sample(frac=1, random_state=123)
            df_data = filter_artefacts(df_data)
            print("Corpus shape: ", df_data.shape)

            vocab = Vocab()
            all_data = []
            all_sents = []
            all_sources = []
            source_counts = defaultdict(int)
            doc_counter = defaultdict(int)

            for idx, row in df_data.iterrows():
                chunk = str(row['publisher'])
                doc_counter[chunk] += 1
                po = row['po']
                period = row['period']
                meta = [po, period]
                source_counts[chunk] += 1
                text = row['text']
                lemmatized_text = row['lemmatized_text']

                sents = text.split(' <eos> ')
                lemmatized_sents = lemmatized_text.split(' <eos> ')
                for sent, lemmatized_sent in zip(sents, lemmatized_sents):
                    vocab.add(sent, lemmatized_sent, chunk, meta)
                    all_sents.append(sent)
                all_sources.append(chunk)

            print('Sources in vocab: ', list(set(all_sources)))
            print('Doc counter: ', doc_counter)

            vocab.make_vocab(vocab_path)
            with open(vocab_output, 'wb') as handle:
                dill.dump(vocab, handle)

            print('Done building vocab.')