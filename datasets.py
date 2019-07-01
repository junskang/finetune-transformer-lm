import os
import csv
import numpy as np

from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from collections import namedtuple

seed = 3535999445

def _rocstories(path):
    with open(path) as f:
        f = csv.reader(f)
        st = []
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = ' '.join(line[1:5])
                c1 = line[5]
                c2 = line[6]
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1])-1)
        return st, ct1, ct2, y

def rocstories(data_dir, n_train=1497, n_valid=374):
    storys, comps1, comps2, ys = _rocstories(os.path.join(data_dir, 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'))
    teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(storys, comps1, comps2, ys, test_size=n_valid, random_state=seed)
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)


PostModifier = namedtuple("PostModifier", ['sent', 'entity', 'pm', 'sent_full', 'wiki_id', 'prev_sent', 'next_sent', 'fileinfo'])
WikiEntity = namedtuple("WikiEntity", ['id', 'label', 'aliases', 'description', 'claims'])

def load_post_modifiers(filepath):
    pms = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            pms.append(PostModifier(*line.split('\t')))
            # if len(pms) < 3:
            #   print (pms[-1])
    print("Loaded [%s]: %d" % (filepath, len(pms)))
    return pms

def load_wiki_data(filepath):
    wiki_data = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            tokens = line.split('\t')
            wiki_data.append(WikiEntity(id=tokens[0],
                                        label=tokens[1],
                                        aliases=tokens[2],
                                        description=tokens[3],
                                        claims=eval(tokens[4])))

    print("Loaded [%s]: %d" % (filepath, len(wiki_data)))
    return wiki_data

class PoMoDelimiters():
    CONTEXT_START = "<context_s>"
    CONTEXT_MIDDLE = "<context_in>"
    CONTEXT_END = "<context_e>"

    CLAIM_START = "<claim_s>"
    CLAIM_MIDDLE = "<claim_in>"
    CLAIM_END = "<claim_e>"

    QUALIFIER_START = "<qualifier_s>"
    QUALIFIER_MIDDLE = "<qualifier_in>"
    QUALIFIER_END = "<qualifier_e>"

    delims = ['CONTEXT_START', 'CONTEXT_MIDDLE', 'CONTEXT_END', 'CLAIM_START', 'CLAIM_MIDDLE', 'CLAIM_END',
              'QUALIFIER_START', 'QUALIFIER_MIDDLE', 'QUALIFIER_END', ]

def listClaims(claims):
    def claim_to_str(property, marker="<claim_in>"):
        return "%s %s %s" % (property[0], marker, property[1])

    claims_str_list = []
    for claim in claims:
        property = claim['property']
        qualifiers = claim['qualifiers']

        claim_str_list = []
        claim_str_list.append(PoMoDelimiters.CLAIM_START)
        claim_str_list.append(claim_to_str(property, marker=PoMoDelimiters.CLAIM_MIDDLE))
        for qualifier in qualifiers:
            claim_str_list.append(PoMoDelimiters.QUALIFIER_START)
            claim_str_list.append(claim_to_str(qualifier, marker=PoMoDelimiters.QUALIFIER_MIDDLE))
            claim_str_list.append(PoMoDelimiters.QUALIFIER_END)

        claim_str_list.append(PoMoDelimiters.CLAIM_END)
        claims_str_list.append(" ".join(claim_str_list))
    return claims_str_list


def printInfoPM(pm):
    fields = pm._fields
    print("- PM")
    for f in fields:
        print("    - %15s: %s" % (f, getattr(pm, f)))


def _pomo(path):
    instances = load_post_modifiers(path)
    wiki_entities = load_wiki_data("%s.wiki" % (path))
    claims_by_id = {e.id: e.claims for e in wiki_entities}

    contexts = []
    claims = []
    pms = []

    for instance in instances:
        instance_claims = claims_by_id[instance.wiki_id]

        contexts.append("%s %s %s %s %s" % (
        PoMoDelimiters.CONTEXT_START, instance.prev_sent, PoMoDelimiters.CONTEXT_MIDDLE, instance.sent,
        PoMoDelimiters.CONTEXT_END))
        claims.append(PoMoDelimiters.CLAIM_MIDDLE.join(listClaims(instance_claims)))
        pms.append(instance.pm)

    return contexts, claims, pms


def pomo(data_dir):
    # tr_contexts, tr_claims, tr_pms = _pomo(os.path.join(data_dir, 'train'))
    # va_contexts, va_claims, va_pms = _pomo(os.path.join(data_dir, 'valid'))
    # te_contexts, te_claims, te_pms = _pomo(os.path.join(data_dir, 'test'))

    va_contexts, va_claims, va_pms = _pomo(os.path.join(data_dir, 'valid'))
    va_contexts, va_claims, va_pms = va_contexts[:100], va_claims[:100], va_pms[:100]
    tr_contexts, tr_claims, tr_pms = va_contexts, va_claims, va_pms
    te_contexts, te_claims, te_pms = va_contexts, va_claims, va_pms

    return (tr_contexts, tr_claims, tr_pms), (va_contexts, va_claims, va_pms), (te_contexts, te_claims)
