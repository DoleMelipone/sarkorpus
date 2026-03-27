from sklearn.feature_extraction.text import CountVectorizer
import dill
import argparse
import plotly.graph_objects as go
from nltk import pos_tag
import numpy as np
import os
import re

from collections import Counter, defaultdict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from french_stopwords import french_stopwords


def get_target_words(input_path, sort_column, treshold=100):
    df = pd.read_csv(input_path, encoding='utf8', sep=';')
    df = df.sort_values(by=[sort_column], ascending=False)
    words = df['word'].tolist()
    for word in words:
        if len(word) <= 1:
            words.remove(word)
    words_short = words[:treshold]
    return words_short


def get_clusters_sent(target, threshold_size_cluster, labels, sentences, id2sents, corpus_slices, docs_folder):

    labels = dill.load(open(labels, 'rb'))
    sentences = dill.load(open(sentences, 'rb'))
    id2sents = dill.load(open(id2sents, 'rb'))

    cluster_to_sentence = defaultdict(lambda: defaultdict(list))
    for cs in corpus_slices:
        for label, sents in zip(labels[target][cs], sentences[target][cs]):
            for sent in sents:
                sent_id = int(str(corpus_slices.index(cs) + 1) + str(sent))
                sent = id2sents[sent_id]
                cluster_to_sentence[label][cs].append(sent)

    counts = {cs: Counter(labels[target][cs]) for cs in corpus_slices}
    all_labels = []
    for slice, c in counts.items():
        slice_labels = [x[0] for x in c.items()]
        all_labels.extend(slice_labels)
    all_labels = set(all_labels)
    all_counts = []
    for l in all_labels:
        all_count = 0
        for slice in corpus_slices:
            count = counts[slice][l]
            all_count += count
        all_counts.append((l, all_count))
    sorted_counts = sorted(all_counts, key=lambda x: x[1], reverse=True)
    sentences = []
    lemmas = []
    metas = []
    labels = []
    categs = []

    for label, count in sorted_counts:
        #print("\n================================")
        #print("Cluster label: ", label, " - Cluster size: ", count)
        if count > threshold_size_cluster:
            for cs in corpus_slices:
                for (sent, lemma, meta) in cluster_to_sentence[label][cs]:
                    sent_clean = sent.strip()
                    lemma_clean = lemma.replace('_<ner>', '').strip()
                    lemma_clean = re.sub(r'_\w+\b', '', lemma_clean).strip()
                    #if sent_clean not in set(sentences):
                    sentences.append(sent_clean)
                    lemmas.append(lemma_clean)
                    labels.append(label)
                    categs.append(cs)
                    metas.append(meta)
                    #print(sent_clean)
        else:
            print("Cluster", label, "is too small - deleted!")

    po = [x[0] for x in metas]
    topic = [x[1] for x in metas]
    #dates = [x[2] for x in metas]
    #publication_type = [x[1] for x in metas]
    sent_df = pd.DataFrame(list(zip(po, topic, categs, labels, sentences, lemmas)),
            columns=['po', 'topic', 'newspaper', 'cluster_label', 'sentence', 'lemmatized_sent'])
    sent_df.to_csv(os.path.join(docs_folder, target + '_sentences.tsv'), encoding='utf-8', index=False, sep='\t')
    return sent_df


def output_distrib(data, word, keyword_clusters, image_folder, translator):
    distrib = data.groupby(['newspaper', "cluster_label"]).size().reset_index(name="count")
    pivot_distrib = distrib.pivot(index='newspaper', columns='cluster_label', values='count')
    pivot_distrib_norm = pivot_distrib.div(pivot_distrib.sum(axis=1), axis=0)
    pivot_distrib_norm = pivot_distrib_norm.fillna(0)
    first_column = pivot_distrib_norm.columns[0]
    #order = list(str(x) for x in pivot_distrib_norm[first_column].keys())
    #order = ['left', 'right']
    print(pivot_distrib_norm)
    order = ['political_0', 'political_1', 'judicial_0', 'judicial_1']
    columns = []
    final_data = []
    #print('Translations:')
    for i in keyword_clusters:
        legend = ", ".join(keyword_clusters[i][:3])#before 7
        if len(legend) > 120:
            legend = ", ".join(legend[:120].split(', ')[:-1])
        if translator is not None:
            print(translator.translate(", ".join(keyword_clusters[i]), src='fr', dest='en').text)
            legend = translator.translate(legend, src='fr', dest='en').text
        name = "Cluster " + str(i) + ": " + legend
        distrib = np.array(list(pivot_distrib_norm[i].fillna(0).array))
        final_data.append((name, distrib))

    final_data = sorted(final_data, reverse=True, key=lambda x:sum(x[1]))
    if len(final_data) <= 10:
        for name, distrib in final_data:
            columns.append(go.Bar(name=name, x=order, y=distrib))
    else:
        for name, distrib in final_data[:9]:
            columns.append(go.Bar(name=name, x=order, y=distrib))
        other_data = final_data[9:]
        other = None
        print(other_data)
        for name, distrib in other_data:
            if other is None:
                other = distrib
                print(distrib, other)
            else:
                other += distrib
                print(distrib, other)

        print('Other: ', other)
        columns.append(go.Bar(name='Other', x=order, y=other))

    fig = go.Figure(data=columns)
    fig.update_layout(
        margin=dict(l=15, r=20, t=20, b=20),
        #width=1200,
        width=600,
        #height=800,
        height=600,
        barmode='stack',
        title='',
        xaxis_title="Newspapers & period",
        yaxis_title="Usage distribution",
        legend_title="",
        font=dict(
            #size=14,
            size=22,
            color="Black"
        ),
        legend = dict(
            yanchor="top",
            y=1.4,
            xanchor="left",
            x=0.3,
            font=dict(
                #size=5,
                size=10,
                color="Black"
            ),
        )
    )
    #fig.show()
    fig.write_image(os.path.join(image_folder, f"{word}.png"))

    return pivot_distrib_norm


def extract_topn_from_vector(feature_names, sorted_items, topn):
    """get the feature names and tf-idf score of top n items"""
    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    return results

def extract_keywords(target_word, word_clustered_data, max_df, topn, lang, docs_folder):
    sw = french_stopwords
    # get groups of sentences for each cluster
    l_sent_clust_dict = defaultdict(list)
    sent_clust_dict = defaultdict(list)
    for i, row in word_clustered_data.iterrows():
        sent = " ".join(row['sentence'].split())
        lemmatized_sent = " ".join(row['lemmatized_sent'].split())
        l_sent_clust_dict[row['cluster_label']].append((sent, lemmatized_sent))

    for label, data in l_sent_clust_dict.items():
        original_sents = "\t".join([x[0] for x in data])
        lemmas = "\t".join([x[1] for x in data])
        sent_clust_dict[label] = (original_sents, lemmas)

    labels = []
    lemmatized_clusters = []
    for label, (sents, lemmatized_sents) in sent_clust_dict.items():
        labels.append(label)
        #lemmatize
        lemmatized_clusters.append(lemmatized_sents)
        #lemmatized_clusters.append(sents)
    #print("Lemmatized clusters: ", lemmatized_clusters)

    # print(list(cv.vocabulary_.keys())[:10])
    tfidf_transformer = TfidfVectorizer(smooth_idf=True, use_idf=True, ngram_range=(1,2), max_df=max_df, max_features=10000)
    tfidf_transformer.fit(lemmatized_clusters)
    feature_names = tfidf_transformer.get_feature_names_out()

    keyword_clusters = {}

    for label, lemmatized_cluster in zip(labels, lemmatized_clusters):
        # generate tf-idf
        tf_idf_vector = tfidf_transformer.transform([lemmatized_cluster])
        # sort the tf-idf vectors by descending order of scores
        tuples = zip(tf_idf_vector.tocoo().col, tf_idf_vector.tocoo().data)
        sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
        # extract only the top n
        keywords = extract_topn_from_vector(feature_names, sorted_items, topn*10)
        keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)


        # filter unigrams that appear in bigrams and remove duplicates
        scores = {x[0]:x[1] for x in keywords}
        #print(scores)

        already_in = set()
        filtered_keywords = []
        for kw, score in keywords:
            if len(kw.split()) == 1:
                for k, s in scores.items():
                    if kw in k and len(k.split()) > 1:
                        if score > s:
                            already_in.add(k)
                        else:
                            already_in.add(kw)
            if len(kw.split()) == 2:
                for k, s in scores.items():
                    if kw in k and len(k.split()) > 2:
                        if score > s:
                            already_in.add(k)
                        else:
                            already_in.add(kw)

            if kw not in already_in and kw != target_word:
                filtered_keywords.append(kw)
                already_in.add(kw)

        keyword_clusters[label] = filtered_keywords[:topn*10]

    final_keywords = {}
    all_data = []
    for c, keywords in keyword_clusters.items():
        sents = sent_clust_dict[c][0].split('\t')
        lemmas = sent_clust_dict[c][1].split('\t')
        all_sents = " ".join(sents)
        all_lemmas = " ".join(lemmas)
        set_lemmatized_sents = set(lemmas)
        #print(set_lemmatized_sents)
        set_sents = set(sents)
        filtered_keywords = []
        for kw in keywords:
            stop = 0
            for word in kw.split():
                if word in sw:
                    stop += 1
            if stop / float(len(kw.split())) < 0.5:
                 num_appearances = 0
                 for sent in set_lemmatized_sents:
                     if kw in sent:
                         num_appearances += 1
                 #print(kw, num_appearances)
                 if num_appearances > 0:
                     if len(kw) > 2:
                         if kw + ' ' + target_word in all_lemmas:
                             kw = kw + ' ' + target_word
                         elif target_word + ' ' + kw in all_lemmas:
                             kw = target_word + ' ' + kw
                         filtered_keywords.append(kw)


        if len(filtered_keywords) == 0:
            filtered_keywords.append('other')
        final_keywords[c] = filtered_keywords[:topn]
        all_data.append((c, ";".join(filtered_keywords[:50]), all_sents))
    return final_keywords



def full_analysis(word, labels, sentences, id2sent, corpus_slices, image_folder, docs_folder, max_df=0.7, topn=15, threshold_size_cluster=5, lang='sl', translator=None):
    clusters_sents_df = get_clusters_sent(word, threshold_size_cluster, labels, sentences, id2sent, corpus_slices, docs_folder)
    keyword_clusters = extract_keywords(word, clusters_sents_df, topn=topn, max_df=max_df, lang=lang, docs_folder=docs_folder)
    for k in keyword_clusters:
        keywords = keyword_clusters[k]
        print(keywords)
    output_distrib(clusters_sents_df, word, keyword_clusters, image_folder, translator)
    return keyword_clusters


def loadData(labels_path, sentences_path, id2sents_path):
    labels = dill.load(open(labels_path, 'rb'))
    sentences = dill.load(open(sentences_path, 'rb'))
    id2sents = dill.load(open(id2sents_path, 'rb'))
    return labels, sentences, id2sents


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure semantic shift')
    parser.add_argument('--method', type=str, default='K5', help='Method to use to select the results for analysis: K5, K7 or AP.')
    args = parser.parse_args()

    if args.method == 'K7':
        method = 'kmeans_7'
    elif args.method == 'K5':
        method = 'kmeans_5'
    elif args.method == 'AP':
        method = 'aff_prop'

    lang = 'fr'
    translator = None

    print("Selecting keywords:")

    interpretation_folder = os.path.join('by_period', 'interpretation')
    if not os.path.exists(interpretation_folder):
        os.makedirs(interpretation_folder)

    results_folder = os.path.join('by_period', 'results')

    corpus_slices = [
        'political_0', 'political_1', 'judicial_0', 'judicial_1'
    ]

    image_folder = os.path.join(interpretation_folder, "images")
    docs_folder = os.path.join(interpretation_folder, "meta")
    target_words = get_target_words(os.path.join(results_folder, 'results_pretrained_selected_words.csv'), 'JSD K5 Avg', treshold=10)

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    if not os.path.exists(docs_folder):
        os.makedirs(docs_folder)

    labels = os.path.join(results_folder, method + "_labels_pretrained.pkl")
    sentences = os.path.join(results_folder, "sents_pretrained.pkl")
    id2sent = os.path.join(results_folder, "id2sents_pretrained.pkl")

    for word in target_words:
        print('##########################################', word, '################################')
        keyword_clusters = full_analysis(word, labels, sentences, id2sent, corpus_slices, image_folder, docs_folder, topn=5, lang=lang, translator=translator)
