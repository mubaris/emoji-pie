from annoy import AnnoyIndex
import rocksdb
import numpy as np
import io
import json

db = rocksdb.DB("fastText.db", rocksdb.Options(create_if_missing=False))
emojiDB = rocksdb.DB("emojiFastText.db", rocksdb.Options(create_if_missing=False))

with io.open('emojiData.json', encoding='utf8') as f:
    data = json.load(f)

f = 300
t = AnnoyIndex(f, metric='angular')

for i, e in enumerate(data):
    j = e["emoji"]
    X = np.frombuffer(emojiDB.get(j.encode()))
    t.add_item(i, X)

t.build(100)
t.save('emojis.ann')

