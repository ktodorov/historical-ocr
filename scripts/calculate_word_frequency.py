import os
from typing import Counter
import nltk

language = 'nl'

icdar_path = os.path.join('data', 'newseye')
icdar_2017_path = os.path.join(icdar_path, '2017', 'full')
icdar_2019_path = os.path.join(icdar_path, '2019', 'full')

paths = []
if language == 'eng' or language == 'fr':
    paths.extend([
        os.path.join(icdar_2017_path, f'{language}_monograph'),
        os.path.join(icdar_2017_path, f'{language}_periodical')
    ])

paths.extend([
    os.path.join(icdar_2019_path, language[:2])
])

if language == 'nl':
    paths = [
        os.path.join(icdar_2019_path, language[:2], 'NL1')
    ]

labels = ['NOUN', 'VERB' ]

words_set = Counter()

for path in paths:
    txt_files = os.listdir(path)
    for txt_file in txt_files:
        txt_file_path = os.path.join(path, txt_file)
        with open(txt_file_path, 'r', encoding='utf8') as file:
            words = file.read().lower().split()
            for word in words:
                words_set[word] += 1

sorted_word_pairs = sorted(words_set.items(), key=lambda x: x[1], reverse=True)
# sorted_words = [x[0] for x in sorted_word_pairs]

a = ['man', 'jaar', 'tijd', 'mensen', 'dag', 'kinderen', 'hand', 'huis', 'dier', 'afbeelding', 'werk', 'naam', 'groot', 'kleine']

print([x for x in sorted_word_pairs if x[0] in a])

# pos_tags = [nltk.pos_tag([word], tagset='universal')[0] for word in sorted_words[:250]]
# remaining_pos_tags = [x[0] for x in pos_tags if x[1] in labels]
# print(remaining_pos_tags[50:100])