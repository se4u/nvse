import pandas as pd
import sys, os

def make_df(results, TRANSLATION):
  df = pd.DataFrame(columns=["system", "nents", "nsents", 
                             "first_place", "second_place", "third_place"]) #"system"
  keys_set = set()
  tmp_res = {}
  for pair, res in results.items():
    nsents = pair[0]
    nents = pair[1]
    if nsents == 0:
      nsents = "1"
    elif nsents == 1:
      nsents = "2-10"
    elif nsents == 2:
      nsents = "11-100"
    else:
      nsents = ">100"
    for pair2, score in res.items():
      sys = pair2[0]
      place = pair2[1]
      tmp_res[(nsents, nents, sys, place)] = score
      keys_set.add((nsents, nents, sys))

  for key in keys_set:
    scores = {}
    for i in [1,2,3]:
      scores[i] = tmp_res[(key[0], key[1], key[2], i)]
  
    df = df.append({"system": TRANSLATION[key[2]], "nents": key[1], "nsents": key[0], \
                    "first_place": scores[1], "second_place": scores[2],
                    "third_place": scores[3]}, ignore_index=True)
  return df

def concat(s, frag):
  return '\n'.join([a + b for (a,b) in zip(s.split('\n'), frag.split('\n'))])

def process(s, i, f):
  return '&'.join([f(e) if idx == i else e for idx, e in enumerate(s.split('&'))])

def get_res_from_file(fname):
  df = pd.read_csv(fname) 
  nver, bm25, bs = [],[],[]
  # print fname
  bm25 = df['Answer.BM25_ranking']
  bs = df['Answer.BS_ranking']
  nver = df['Answer.NVER_ranking']

  for i in range(len(nver)):
    if nver[i] == bs[i] or nver[i] == bm25[i] or bs[i] == bm25[i]:
      print i+2, fname

  results = {}
  systems = {"nver": nver, "bm25": bm25, "bs": bs}
  for name,sys in systems.items():
    for _ in range(1,4):
      results[name, _] = len([x for x in sys if x == _])
  return results, ((nver == 1).sum() >= (bm25 == 1).sum())

def main():
  TRANSLATION = dict(bm25='SetEx', bs='W2Vec', nver='NVGE')
  results = {}
  beats = {}
  for binid in range(3):
    for nents in [3,5]:
      fn = 'hit_setex_w2v_nvge_binid~%d_nent~%d_results.csv'%(binid,nents)
      results[(binid, nents)], beats[(binid, nents)] = get_res_from_file(fn)
  print beats
  df = make_df(results, TRANSLATION).sort_values(by='nents nsents system'.split())
  print df.pivot_table(aggfunc='first', values='first_place second_place third_place'.split(), 
                       index=['nents', 'nsents'], 
                       columns='system').to_latex()
  TRANSLATION = dict(bm25='BM25', bs='BaySe', nver='NVGE')
  results = {}
  beats = {}
  for binid in range(3):
    for nents in [3,5]:
      fn = 'hit_binid~%d_nent~%d_results.csv'%(binid,nents)
      results[(binid, nents)], beats[(binid, nents)] = get_res_from_file(fn)
  print beats
  df = make_df(results, TRANSLATION).sort_values(by='nents nsents system'.split())
  print df.pivot_table(aggfunc='first', values='first_place second_place third_place'.split(),
                       index=['nents', 'nsents'], columns='system').to_latex()
  return


if __name__ == '__main__':
  main()
