import json
import os
from collections import Counter
from fastcoref import FCoref, LingMessCoref

pronoms = {'il', 'Il', 'ils', 'lui', 'elle', 'elles', 'moi', 'me'}

def save_data(file, ans, papers):
    with open(file) as f:
        data = json.load(f)
    all_years = {y: {} for y in ans}
    for v in data.values():
        year = v['annee']
        source = v['journal_clean']
        if year in ans and source in papers:
            #if source == "l'Humanité":
            #    source = "L'Humanité"
            if source not in all_years[year]:
                all_years[year][source] = []
            all_years[year][source].append(v['texte'])
    """for paper, text in all_years[2010].items():
        path = os.path.join('coref_resolution', '2010')
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path, paper+".txt"), "w") as f:
            f.writelines(text)"""
    return all_years

def pred_one_paper(preds, n):
    sarkoref = []
    for i in range(n):
        clusters = preds[i].get_clusters(as_strings=True)
        for chain in clusters:
            for occ in chain:
                if 'Nicolas Sarkozy' == occ:
                    sarkoref += chain
                    break
    sarkoref = Counter(sarkoref)
    return sarkoref

def coref_by_year(data, outpath):
    model = FCoref(device='cpu', nlp='fr_core_news_lg')
    res = {}
    for y,v in data.items():
        res[y] = {}
        for j,v2 in v.items():
            n = len(v2)
            preds = model.predict(v2)
            sarkoref = pred_one_paper(preds, n)
            res[y][j] = dict(sarkoref)
        with open(os.path.join(outpath, str(y)+'.json'), 'w') as jfile:
            json.dump(res[y], jfile)
    return res


def split_long_text(text, max_len=1200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_len):
        chunks.append(" ".join(words[i:i+max_len]))
    return chunks

def pred_one_paper_v2(preds):
    sarkoref = Counter()
    for pred in preds:
        clusters = pred.get_clusters(as_strings=True)
        for chain in clusters:
            if "Nicolas Sarkozy" in chain:
                sarkoref.update(chain)
    return sarkoref

def coref_by_year_v2(data, outpath, batch_size=8, device="cpu"):
    model = LingMessCoref(
        device=device,
        nlp="fr_core_news_lg"
    )
    res = {}
    for y, v in data.items():
        res[y] = {}
        for j, texts in v.items():
            expanded_texts = []
            for t in texts:
                expanded_texts.extend(split_long_text(t))

            sarkoref_total = Counter()
            # batch prediction
            for i in range(0, len(expanded_texts), batch_size):
                batch = expanded_texts[i:i+batch_size]
                preds = model.predict(texts=batch)
                sarkoref_total.update(pred_one_paper_v2(preds))
            res[y][j] = dict(sarkoref_total)

        filepath = os.path.join(outpath, f"{y}.json")
        with open(filepath, "w") as jfile:
            json.dump(res[y], jfile)

    return res
    
if __name__ == '__main__':
    file = os.path.join('datasets', 'sarko_affaires.json')
    out = os.path.join('coref_resolution', 'all_corefs_fr')
    ans = range(2002, 2003)
    papers = ['Le Figaro', 'Libération', 'Le Monde']
    data = save_data(file, ans, papers)
    res = coref_by_year(data, out)

    """for k,v in res.items():
        print(k)
        print(v)
        print()"""