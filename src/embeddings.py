import torch
import gensim

from src.vocabulary import Vocabulary


def get_embeddings(vocabulary: Vocabulary, embeddings_filename, embeddings_dim, binary=False):
    embeddings = torch.div(torch.randn(vocabulary.size(), embeddings_dim), 100)
    w2v = gensim.models.KeyedVectors.load_word2vec_format(embeddings_filename, binary=binary)
    unknown_words_count = 0
    for i, word in enumerate(vocabulary.index2word):
        if word in w2v:
            embeddings[i] = torch.FloatTensor(w2v[word])
        else:
            unknown_words_count += 1
    embeddings[0] = torch.zeros((embeddings_dim,))
    print("Unknown words in {filename}: {count}".format(
        count=unknown_words_count,
        filename=embeddings_filename
    ))
    return embeddings


def shrink_w2v(input_filename, vocabulary, found_border, output_filename, print_step=100000):
    vocabulary_embeddings = dict()
    all_count = 0
    found_count = 0
    words = set(vocabulary.word2index.keys())
    with open(input_filename, "r", encoding='utf-8', errors='ignore') as r:
        line = next(r)
        dimension = int(line.strip().split()[1])
        for line in r:
            if all_count % print_step == 0:
                print("Parsed words: {}, intersection: {}, unknown words:{}".format(
                    all_count, found_count, len(words)))
            if found_count == found_border:
                break
            is_success_parse = parse_line(line, vocabulary, words, vocabulary_embeddings)
            found_count += int(is_success_parse)
            all_count += 1
        vocabulary_embeddings = {key:value for key, value in vocabulary_embeddings.items()
                                 if len(value) == dimension}
        with open(output_filename, "w", encoding='utf-8') as w:
            w.write(str(len(vocabulary_embeddings.items())) + " " + str(dimension) + "\n")
            for i, word in enumerate(vocabulary.index2word):
                if i in vocabulary_embeddings:
                    embedding = " ".join([str(j) for j in list(vocabulary_embeddings[i])])
                    w.write(word + " " + embedding + "\n")
        print("Unknown examples:", list(words)[:100])


def parse_line(line, vocabulary, vocabulary_words, embeddings):
    try:
        word = line.strip().split()[0]
        if word in vocabulary_words:
            embedding = [float(i) for i in line.strip().split()[1:]]
            vocabulary_words.remove(word)
            embeddings[vocabulary.word2index[word]] = embedding
            return True
    except ValueError:
        return False
    return False
