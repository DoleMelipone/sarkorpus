#https://fr.wikipedia.org/wiki/Similarit%C3%A9_cosinus
#https://fr.wikipedia.org/wiki/Indice_et_distance_de_Jaccard

import dill
import pandas as pd
import ot
from scipy.spatial.distance import cdist

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from collections import Counter
from scipy.stats import entropy
from collections import defaultdict
import numpy as np
import os
import re


def get_target_words(input_path):
    df = pd.read_csv(input_path, encoding='utf8', sep=';')
    words = df['word'].tolist()
    return words


def combine_clusters(labels, embeddings, treshold=10, remove=[]):
    #print("Begin", Counter(labels))
    cluster_embeds = defaultdict(list)
    for label, embed in zip(labels, embeddings):
        cluster_embeds[label].append(embed)
    #print("Counts: ", Counter(labels))
    #print("Num clusters: ", len(set(labels)), remove)
    min_num_examples = treshold
    legit_clusters = []
    for id, num_examples in Counter(labels).items():
        if num_examples >= treshold:
            legit_clusters.append(id)
        if id not in remove and num_examples < min_num_examples:
            min_num_examples = num_examples
            min_cluster_id = id

    if len(set(labels)) == 2:
        return labels

    min_dist = 1
    all_dist = []
    cluster_labels = ()
    embed_list = list(cluster_embeds.items())
    for i in range(len(embed_list)):
        for j in range(i+1,len(embed_list)):
            id, embed = embed_list[i]
            id2, embed2 = embed_list[j]
            if id in legit_clusters and id2 in legit_clusters:
                #dist = compute_pairwise_embedding_dist(embed, embed2)
                dist = compute_averaged_embedding_dist(embed, embed2)
                all_dist.append(dist)
                if dist < min_dist:
                    min_dist = dist
                    cluster_labels = (id, id2)

    std = np.std(all_dist)
    avg = np.mean(all_dist)
    limit = avg - 2 * std
    if min_dist < limit:
        for n, i in enumerate(labels):
            if i == cluster_labels[0]:
                labels[n] = cluster_labels[1]
        return combine_clusters(labels, embeddings, treshold, remove)

    if min_num_examples >= treshold:
        #print("Final", Counter(labels))
        return labels


    min_dist = 2
    cluster_labels = ()
    for id, embed in cluster_embeds.items():
        if id != min_cluster_id:
            dist = compute_averaged_embedding_dist(embed, cluster_embeds[min_cluster_id])
            if dist < min_dist:
                min_dist = dist
                cluster_labels = (id, min_cluster_id)

    if cluster_labels[0] not in legit_clusters:
        for n, i in enumerate(labels):
            if i == cluster_labels[0]:
                labels[n] = cluster_labels[1]
    else:
        limit = avg - std
        if min_dist < limit:
            for n, i in enumerate(labels):
                if i == cluster_labels[0]:
                    labels[n] = cluster_labels[1]
        else:
            remove.append(min_cluster_id)
    return combine_clusters(labels, embeddings, treshold, remove)


def compute_inner_cluster_dist(embeddings, k=2):
    if len(embeddings) < k:
        return 0
    labels, centroids = cluster_word_embeddings_k_means(embeddings, k, random_state)
    cluster_embeds = defaultdict(list)
    for label, embed in zip(labels, embeddings):
        cluster_embeds[label].append(embed)
    all_dist = []
    for id, embed in cluster_embeds.items():
        for id2, embed2 in cluster_embeds.items():
            if id != id2:
                dist = compute_pairwise_embedding_dist(embed, embed2)
                all_dist.append(dist)
    return sum(all_dist)/len(all_dist)



def compute_jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def custom_measure(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    return (abs(p-q).sum()) / (abs(p+q).sum())


def filter_english(text, word):
    if word in text and word[:-3] not in text.split():
        return False
    else:
        print(text, word)
        return True


def cluster_word_embeddings_aff_prop(word_embeddings):
    clustering = AffinityPropagation().fit(word_embeddings)
    labels = clustering.labels_
    counts = Counter(labels)
    #print("Aff prop num of clusters:", len(counts))
    exemplars = clustering.cluster_centers_
    return labels, exemplars


def cluster_word_embeddings_dbscan(word_embeddings):
    clustering = DBSCAN().fit(word_embeddings)
    labels = clustering.labels_
    counts = Counter(labels)
    #print("DBSCAN num of clusters:", len(counts))
    return labels


def cluster_word_embeddings_k_means(word_embeddings, k, random_state):
    clustering = KMeans(n_clusters=k, random_state=random_state).fit(word_embeddings)
    labels = clustering.labels_
    exemplars = clustering.cluster_centers_
    return labels, exemplars


def compute_averaged_embedding_dist(t1_embeddings, t2_embeddings):
    t1_mean = np.mean(t1_embeddings, axis=0)
    t2_mean = np.mean(t2_embeddings, axis=0)
    dist = 1.0 - cosine_similarity([t1_mean], [t2_mean])[0][0]
    #print("Averaged embedding cosine dist:", dist)
    return dist

def compute_pairwise_embedding_dist(t1_embeddings, t2_embeddings):
    t1_mean = np.mean(t1_embeddings, axis=0)
    df_cs = 1.0 - cosine_similarity([t1_mean], t2_embeddings)
    return np.average(df_cs)

def compute_divergence_from_cluster_labels(embeds1, embeds2, labels1, labels2, weights1, weights2, treshold):
    label_list_1 = sorted(list(set(list(labels1))))
    label_list_2 = sorted(list(set(list(labels2))))
    labels_all = list(np.concatenate((labels1, labels2)))
    n_senses = set(labels_all)
    counts1 = Counter(labels1)
    counts2 = Counter(labels2)
    #print(counts1)
    #print(counts2)
    not_enough_elements = set()
    missing_elements_weights_1 = 0
    missing_elements_weights_2 = 0

    #check legit clusters
    for l, c in zip(labels1, weights1):
        if counts1[l] + counts2[l] <= treshold:
            missing_elements_weights_1 += c
            not_enough_elements.add(l)
    for l, c in zip(labels2, weights2):
        if counts1[l] + counts2[l] <= treshold:
            missing_elements_weights_2 += c
            not_enough_elements.add(l)
    for el in not_enough_elements:
        n_senses.remove(el)

    #generate distributions
    t1 = defaultdict(int)
    for l, c in zip(labels1, weights1):
        if counts1[l] + counts2[l] > treshold:
            t1[l] += c
    not_in_t1 = list(n_senses - set(t1.keys()))
    for l in not_in_t1:
        t1[l] = 0
    t1 = sorted(t1.items(), key=lambda x: x[0])
    t1_dist = np.array([x[1] / (sum(weights1) - missing_elements_weights_1) for x in t1])

    t2 = defaultdict(int)
    for l, c in zip(labels2, weights2):
        if counts1[l] + counts2[l] > treshold:
            t2[l] += c
    not_in_t2 = list(n_senses - set(t2.keys()))
    for l in not_in_t2:
        t2[l] = 0
    t2 = sorted(t2.items(), key=lambda x: x[0])
    t2_dist = np.array([x[1] / (sum(weights2) - missing_elements_weights_2) for x in t2])

    #print("Distrib 1", t1_dist)
    #print("Distrib 2", t2_dist)
    if len(t1_dist) == 0 or len(t2_dist) == 0:
        return 0, 0

    shape = embeds1[0].shape
    emb1_means = np.array([np.mean(embeds1[labels1 == clust], 0) if clust in label_list_1 else np.zeros(shape) for clust in n_senses])
    emb2_means = np.array([np.mean(embeds2[labels2 == clust], 0) if clust in label_list_2 else np.zeros(shape) for clust in n_senses])
    M = np.nan_to_num(np.array([cdist(emb1_means, emb2_means, metric='cosine')])[0], nan=1)

    wass = ot.emd2(t1_dist, t2_dist, M)
    jsd = compute_jsd(t1_dist, t2_dist)
    return jsd, wass

def detect_meaning_gain_and_loss(labels1, labels2, treshold):
    labels1 = list(labels1)
    labels2 = list(labels2)
    all_count = Counter(labels1 + labels2)
    first_count = Counter(labels1)
    second_count = Counter(labels2)
    gained_meaning = False
    lost_meaning = False
    all = 0
    meaning_gain_loss = 0

    for label, c in all_count.items():
        all += c
        if c >= treshold:
            if label not in first_count or first_count[label] <= 2:
                gained_meaning=True
                meaning_gain_loss += c
            if label not in second_count or second_count[label] <= 2:
                lost_meaning=True
                meaning_gain_loss += c
    #print(name, "gained meaning", gained_meaning, "lost meaning", lost_meaning)
    return str(gained_meaning) + '/' + str(lost_meaning), meaning_gain_loss/all


def compute_divergence_across_many_periods(calculate_diff, embeddings, counts, labels, splits, corpus_slices, treshold, metric):
    all_clusters = []
    all_embeddings = []
    all_counts = []
    clusters_dict = {}
    distrib_dict = {}
    for split_num, split in enumerate(splits):
        if split_num > 0:
            clusters = labels[splits[split_num-1]:split]
            clusters_dict[corpus_slices[split_num - 1]] = clusters
            all_clusters.append(clusters)
            ts_embeds = embeddings[splits[split_num - 1]:split]
            ts_counts = counts[splits[split_num - 1]:split]
            all_embeddings.append(ts_embeds)
            all_counts.append(ts_counts)
            distrib = defaultdict(int)
            for l, c in zip(clusters, ts_counts):
                distrib[l] += c
            distrib = sorted(distrib.items(), key=lambda x: x[0])
            distrib = np.array([x[1] / sum(ts_counts) for x in distrib])
            distrib_dict[corpus_slices[split_num - 1]] = distrib

    all_scores = []
    for (i,j) in calculate_diff:
        jsd, wass = compute_divergence_from_cluster_labels(all_embeddings[i], all_embeddings[j], all_clusters[i],
                                                           all_clusters[j], all_counts[i], all_counts[j], treshold)
        try:
            if metric == 'JSD':
                measure = jsd
            if metric == 'WD':
                measure = wass  # hmean([jsd, wass])
        except:
            measure = 0
        all_scores.append(measure)

    avg_score = sum(all_scores)/len(all_scores)
    all_scores.extend([avg_score])
    all_scores = [float("{:.6f}".format(score)) for score in all_scores]
    return all_scores, clusters_dict, distrib_dict
    #return all_scores, all_meanings, clusters_dict, distrib_dict


if __name__ == '__main__':
    random_state = 123
    treshold = 5
    get_additional_info = True
    metric = 'JSD'

    calculate_diff = [(0, 1)]

    years = [y for y in range(2015, 2026)]
    names1 = {'Libération': 'liberation', 'Le Figaro': 'figaro'}
    names2 = {'Libération': 'liberation', 'Le Monde': 'monde'}
    names3 = {'Le Monde': 'monde', 'Le Figaro': 'figaro'}
    comps = {'libe-fig': names1, 'libe-monde': names2, 'monde-fig': names3}

    for year in years:
        for comp in comps:
            corpus_slices = list(comps[comp].values())
            embeddings_path = os.path.join('by_year_newspapers', comp, str(year), 'embeddings', 'by_year_vocab.pickle')
            results_dir = os.path.join('by_year_newspapers', comp, str(year), 'results')

            target_words = []

            emb_type = 'pretrained'
            embeddings_file = embeddings_path
            print("Loading ", embeddings_file)
            try:
                bert_embeddings, count2sents = dill.load(open(embeddings_file, 'rb'))
            except:
                bert_embeddings = dill.load(open(embeddings_file, 'rb'))
                count2sents = None
            id2sent = {}

            target_words = list(bert_embeddings.keys())
            target_words = target_words[0:100]

            jsd_vec = []
            cosine_dist_vec = []
            sentence_dict = {}

            kmeans_5_labels_dict = {}
            kmeans_5_centroids_dict = {}

            aff_prop_pref = -430
            print("Clustering BERT embeddings")
            print("Len target words: ", len(target_words))

            results = []

            print("Num. words in embeds: ", len(bert_embeddings.keys()))

            rejected = []
            for i, word in enumerate(target_words):
                if not word.endswith('<ner>') and (word.endswith('_NOUN') or word.endswith('_PROPN')):
                    print("\n=======", i + 1, "- word:", word.upper(), "=======")

                    if word not in bert_embeddings:
                        continue
                    try:
                        emb = bert_embeddings[word]
                        if i == 0:
                            print("Time periods in embeds: ", emb.keys())

                        all_embeddings = []
                        all_sentences = {}
                        all_counts = []
                        splits = [0]
                        all_slices_present = True
                        all_freqs = []

                        summed_cs_counts = []

                        for cs in corpus_slices:
                            cs_embeddings = []
                            cs_sentences = []
                            cs_counts = []

                            count_all = 0
                            text_seen = set()

                            if cs not in emb:
                                all_slices_present = False
                                print('Word missing in slice: ', cs)
                                continue

                            counts = [x[1] for x in emb[cs]]
                            summed_cs_counts.append(sum(counts))
                            #print('Counts: ', counts)
                            all_freqs.append(sum(counts))
                            cs_text = cs + '_text'
                            print("Slice: ", cs)
                            print("Num embeds: ", len(emb[cs]))
                            num_sent_codes = 0

                            for idx in range(len(emb[cs])):

                                #get summed embedding and its count, devide embedding by count
                                try:
                                    e, count_emb = emb[cs][idx]
                                    e = e/count_emb
                                except:
                                    e = emb[cs][idx]

                                sents = set()

                                #print("Num sentences: ", len(sent_codes))
                                if count2sents is not None:
                                    sent_codes = emb[cs_text][idx]
                                    num_sent_codes += len(sent_codes)
                                    for sent in sent_codes:
                                        if sent in count2sents[cs]:
                                            sent_data = count2sents[cs][sent]
                                        sent_id = int(str(corpus_slices.index(cs) + 1) + str(sent))
                                        id2sent[sent_id] = sent_data
                                        sents.add(sent)

                                cs_embeddings.append(e)
                                cs_sentences.append(sents)
                                cs_counts.append(count_emb)

                            all_embeddings.append(np.array(cs_embeddings))
                            all_sentences[cs] = cs_sentences
                            all_counts.append(np.array(cs_counts))
                            splits.append(splits[-1] + len(cs_embeddings))


                        print("Num all sents: ", num_sent_codes)
                        print("Num words in corpus slice: ", summed_cs_counts)

                        embeddings_concat = np.concatenate(all_embeddings, axis=0)
                        counts_concat = np.concatenate(all_counts, axis=0)

                        if embeddings_concat.shape[0] < 5 or not all_slices_present:
                            continue
                        else:
                            kmeans_5_labels, kmeans_5_centroids = cluster_word_embeddings_k_means(embeddings_concat, 5, random_state)
                            kmeans_5_labels = combine_clusters(kmeans_5_labels, embeddings_concat, treshold=treshold, remove=[])
                            all_kmeans5_jsds, clustered_kmeans_5_labels, distrib_kmeans_5 = compute_divergence_across_many_periods(calculate_diff, embeddings_concat, counts_concat, kmeans_5_labels, splits, corpus_slices, treshold, metric)
                            all_freqs = all_freqs + [sum(all_freqs)]
                            word = re.sub(r'_\w+\b', '', word)
                            word_results = [word] + all_kmeans5_jsds + all_freqs
                            print("Results:", word_results)
                            results.append(word_results)

                        #add results to dataframe for saving
                        if get_additional_info:
                            sentence_dict[word] = all_sentences

                            kmeans_5_labels_dict[word] = clustered_kmeans_5_labels
                            kmeans_5_centroids_dict[word] = kmeans_5_centroids

                        columns = ['word']
                        #methods = [metric + ' K5', 'MEANING GAIN/LOSS']
                        methods = [metric + ' K5']

                        for method in methods:
                            for cs1, cs2 in calculate_diff:
                                columns.append(method + ' ' + corpus_slices[cs1] + '=>' + corpus_slices[cs2])
                            if method != 'MEANING GAIN/LOSS':
                                columns.append(method + ' Avg')
                        for num_slice, cs in enumerate(corpus_slices):
                            columns.append('FREQ' + ' ' + cs)
                        columns.append('FREQ All')

                        if not os.path.exists(results_dir):
                            os.makedirs(results_dir)

                        if not get_additional_info:
                            csv_file = results_dir + "/results_" + emb_type + "_all_words.csv"
                        else:
                            csv_file = results_dir + "/results_" + emb_type + "_selected_words.csv"

                        # save results to CSV
                        results_df = pd.DataFrame(results, columns=columns)
                        results_df = results_df.sort_values(by=[metric + ' K5 Avg'], ascending=False)
                        results_df.to_csv(csv_file, sep=';', encoding='utf-8', index=False)
                        print("Done! Saved results in", csv_file, "!")

                        if get_additional_info:
                            # save cluster labels and sentences to pickle
                            dicts = [(kmeans_5_centroids_dict, 'kmeans_5_centroids'), (kmeans_5_labels_dict, 'kmeans_5_labels'), (sentence_dict, "sents"), (id2sent, "id2sents")]

                            for data, name in dicts:
                                data_file = os.path.join(results_dir, name + "_" + emb_type + ".pkl")
                                centroids_file = results_dir + "centroids_" + emb_type + ".pkl"
                                pf = open(data_file, 'wb')
                                dill.dump(data, pf)
                                pf.close()
                    except :
                        rejected.append(word)
        
        print(rejected)
