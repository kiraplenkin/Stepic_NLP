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

def get_vocabulary(text):
    word_count = defaultdict(int)
    doc_n = 0

    for text in docs:
        doc_n += 1
        unique_text_tokens = set(text)
        for token in unique_text_tokens:
            word_count[token] += 1

    sorted_word_counts = sorted(
        word_count.items(),
        reverse=False,
        key=lambda pair: (pair[1], pair[0])
    )

    word2id = {word: i for i, (word, _) in enumerate(sorted_word_counts)}

    word2freq = np.array([cnt / doc_n for _, cnt in sorted_word_counts], dtype='float32')

    return word2id, word2freq


word2id, word2freq = get_vocabulary(docs)

for token in word2id:
    print(token, end=' ')

print('\n', *word2freq)
