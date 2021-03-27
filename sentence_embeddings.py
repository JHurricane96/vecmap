import numpy as np
from pathlib import Path
import embeddings

def preprocess(texts_files, vocabs, xp):
  print("Starting preprocessing...")

  sent_id_sets = []
  for lang_id, text_file in enumerate(texts_files):
    sent_ids = set()
    with open(text_file, "r") as f:
      for line in f:
        sent_id, _ = line.strip().split("\t")
        sent_ids.add(sent_id)
    sent_id_sets.append(sent_ids)
    print("Finished sentence ID extraction from language", lang_id)
  sent_ids = set.intersection(*sent_id_sets)
  sent_ids = {sent_id: idx for idx, sent_id in enumerate(sent_ids)}

  all_embs = []
  np.seterr(all="raise")
  for lang_id, text_file in enumerate(texts_files):
    vocab = {word: idx for idx, word in enumerate(vocabs[lang_id])}
    embs = np.zeros((len(vocab), len(sent_ids)), dtype=np.float32)
    with open(text_file, "r") as f:
      for line in f:
        sent_id, sent = line.strip().split("\t")
        sent_idx = sent_ids.get(sent_id, None)
        if sent_idx == None:
          continue
        words = sent.strip().split()
        for word in words:
          if word in vocab:
            embs[vocab[word], sent_idx] += 1
    all_embs.append(xp.asarray(embs))
    print("Finished word count for language", lang_id)

  print("Finised preprocessing")
  return all_embs
