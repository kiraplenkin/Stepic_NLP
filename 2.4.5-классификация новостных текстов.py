import numbers as np
import re

TOKENIZE_RE = re.compile(r'[\w\d]+', re.I)

def tokenize(txt):
    return TOKENIZE_RE.findall(txt)


text = """Казнить нельзя, помиловать. Нельзя наказывать.
Казнить, нельзя помиловать. Нельзя освободить.
Нельзя не помиловать.
Обязательно освободить."""

docs = []

for line in text.split('\n'):
    docs.append(tokenize(line.strip().lower()))

count_tokens = {}
for doc in docs:
    for token in doc:
        if token not in count_tokens:
            count_tokens[token] = 0

for doc in docs:
    for token in count_tokens:
        if token in doc:
            count_tokens[token] += 1

vocabulary = []
for token in count_tokens:
    vocabulary.append({
        'token': token,
        'DF': count_tokens[token] / len(docs)
    })

vocabulary.sort(key=lambda x: (x['DF'], x['token']))

word2id = {}
word2freq = {}
for i, token in enumerate(vocabulary):
    word2id[token['token']] = i
    word2freq[token['token']] = token['DF']

for token in word2id:
    print(token, end=' ')
print()

for token in word2freq:
    print(word2freq[token], end=' ')