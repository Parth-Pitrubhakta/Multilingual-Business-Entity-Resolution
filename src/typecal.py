"""Vocabulary-free pair types, calibrated on training truth (US/India), applied to France.

type = address relation x legal-form relation x street similarity x name relation x country-token flag.
Word classes are label-free per country: noise words (appended after the legal form), typo (edit-similar to
the dropped word or containing digits), then by same-house-number rate: hirate >=0.7, desc 0.3-0.7, decoyw <0.3.
"""
import sys, re

from common import cache, read_ground_truth, read_source
from combine import word_rates
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cpdist

NOISE = {'US': ['center', 'services', 'service', 'partners'], 'India': ['center', 'services', 'service', 'partners'],
         'France': ['services', 'groupe']}
CTOK = {'US': r'\b(usa|america)\b', 'India': r'\bindia\b', 'France': r'\bfr[aâ]nce\b'}
STOP = {'rue', 'ave', 'blvd', 'chem', 'imp', 'pl', 'all', 'rte', 'crs', 'st', 'rd', 'dr', 'ln', 'ct', 'de', 'du', 'des', 'la',
        'le', 'les', 'd', 'l', 'bis', 'ter', 'no', 'n', 'of', 'the', 'road', 'street', 'hwy', 'pkwy', 'cir', 'ter', 'pl', 'way'}


def _street(comps, city):
    if comps is None:
        return ''
    parts = comps.split('|')
    cand = [p for p in parts if re.search(r'\d', p)] or parts[:1]
    toks = [t for t in re.sub(r'\d+', '', cand[0]).split() if t not in STOP and len(t) > 1]
    cityt = set((city or '').split())
    return ' '.join(t for t in toks if t not in cityt)


def typed(split, pairs=None, countries=None):
    cols = ['rec_id', 's1_id', 'hnr_eq', 'ad_tset', 'hn_rel', 'ad_empty2', 'p1']
    f = pl.read_parquet(cache(f'{split}_feats.parquet'), columns=cols)
    if pairs is not None:
        f = f.join(pairs, on=['rec_id', 's1_id'])
    n = pl.read_parquet(cache(f'{split}_norm.parquet'), columns=['entity_id', 'business_name', 'n_core', 'n_legal', 'country', 'a_comps', 'a_city'])
    if countries:
        n1 = n.filter(pl.col('country').is_in(countries))
    else:
        n1 = n
    n = n.with_columns(pl.struct(['a_comps', 'a_city']).map_elements(lambda s: _street(s['a_comps'], s['a_city']), return_dtype=pl.String).alias('street'),
                       pl.col('business_name').str.to_lowercase().alias('lname'))
    s1 = n.select(pl.col('entity_id').alias('s1_id'), 'country', pl.col('n_core').str.split(' ').alias('t1'), pl.col('n_legal').alias('l1'),
                  pl.col('street').alias('st1'), pl.col('lname').alias('ln1'))
    if countries:
        s1 = s1.filter(pl.col('country').is_in(countries))
    rc = n.select(pl.col('entity_id').alias('rec_id'), pl.col('n_core').str.split(' ').alias('t2'), pl.col('n_legal').alias('l2'),
                  pl.col('street').alias('st2'), pl.col('lname').alias('ln2'))
    f = f.join(s1, on='s1_id').join(rc, on='rec_id')
    f = f.with_columns(pl.col('t2').list.set_difference(pl.col('t1')).alias('ex'), pl.col('t1').list.set_difference(pl.col('t2')).alias('mi'))
    f = f.with_columns(pl.col('ex').list.first().alias('w'), pl.col('mi').list.first().alias('d'))
    f = f.join(word_rates(split).select('country', 'w', 'rate'), on=['country', 'w'], how='left')
    nz = pl.concat([pl.DataFrame({'country': c, 'w': ws}) for c, ws in NOISE.items()]).with_columns(pl.lit(1).alias('isnoise'))
    f = f.join(nz, on=['country', 'w'], how='left')
    w = f['w'].fill_null('').to_list(); d = f['d'].fill_null('').to_list()
    sim = cpdist(w, d, scorer=Levenshtein.normalized_similarity, workers=-1)
    st = cpdist(f['st1'].fill_null('').to_list(), f['st2'].fill_null('').to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    f = f.with_columns(pl.Series('wsim', sim), pl.Series('stsim', st))
    # country token count difference (stop word in normalisation)
    ct = []
    for c, rx in CTOK.items():
        ct.append(pl.when(pl.col('country') == c).then(pl.col('ln2').str.count_matches(rx) - pl.col('ln1').str.count_matches(rx)))
    f = f.with_columns(pl.coalesce(ct).fill_null(0).alias('dctok'))
    wcls = (pl.when(pl.col('isnoise') == 1).then(pl.lit('noise'))
            .when(pl.col('w').str.contains(r'\d') | (pl.col('wsim') >= 0.5)).then(pl.lit('typo'))
            .when(pl.col('rate') >= 0.7).then(pl.lit('hirate')).when(pl.col('rate') >= 0.3).then(pl.lit('desc'))
            .when(pl.col('rate').is_not_null()).then(pl.lit('decoyw')).otherwise(pl.lit('rare')))
    nm = (pl.when(pl.col('ex').list.len() == 0).then(pl.when(pl.col('mi').list.len() == 0).then(pl.lit('same')).when(pl.col('mi').list.len() == 1).then(pl.lit('drop1')).otherwise(pl.lit('drop2+')))
          .when(pl.col('ex').list.len() == 1).then(pl.when(pl.col('mi').list.len() == 0).then(pl.lit('add-')).when(pl.col('mi').list.len() == 1).then(pl.lit('swap-')).otherwise(pl.lit('multi-')) + wcls)
          .otherwise(pl.lit('multi')))
    same = (pl.col('hnr_eq') == 1) & (pl.col('ad_tset') >= 95)
    addr = (pl.when(pl.col('ad_empty2') == 1).then(pl.lit('name-only')).when(same).then(pl.lit('same-addr')).when(pl.col('hnr_eq') == 1).then(pl.lit('same-hn'))
            .when(pl.col('hn_rel') == -1).then(pl.lit('hn-miss')).when(pl.col('hn_rel').is_in([1, 2, 3])).then(pl.lit('digit')).when(pl.col('hn_rel').is_in([4, 5])).then(pl.lit('near')).otherwise(pl.lit('other')))
    e1 = pl.col('l1').is_null() | (pl.col('l1') == ''); e2 = pl.col('l2').is_null() | (pl.col('l2') == '')
    legal = pl.when(e1 & e2).then(pl.lit('none')).when(e1 | e2).then(pl.lit('one')).when(pl.col('l1') == pl.col('l2')).then(pl.lit('same')).otherwise(pl.lit('diff'))
    street = pl.when(pl.col('ad_empty2') == 1).then(pl.lit('-')).when((pl.col('st1') == '') | (pl.col('st2') == '')).then(pl.lit('na')).when(pl.col('stsim') >= 85).then(pl.lit('st=')).when(pl.col('stsim') >= 60).then(pl.lit('st~')).otherwise(pl.lit('st!'))
    ctok = pl.when(pl.col('dctok') > 0).then(pl.lit('+C')).when(pl.col('dctok') < 0).then(pl.lit('-C')).otherwise(pl.lit(''))
    f = f.with_columns(addr.alias('addr'), legal.alias('legal'), street.alias('street'), nm.alias('nm'), ctok.alias('ctok'))
    f = f.with_columns(pl.concat_str(['addr', 'legal', 'street', 'nm', 'ctok'], separator='|').alias('type'))
    return f.select('rec_id', 's1_id', 'country', 'addr', 'legal', 'street', 'nm', 'ctok', 'type', 'p1')


if __name__ == '__main__':
    tr = typed('train').join(read_ground_truth().with_columns(pl.lit(1).alias('y')), on=['rec_id', 's1_id'], how='left').with_columns(pl.col('y').fill_null(0))
    tr.write_parquet(cache('typecal_train.parquet'))
    te = typed('test', countries=['France', 'US'])
    te.write_parquet(cache('typecal_test.parquet'))
    print(tr.shape, te.shape)
