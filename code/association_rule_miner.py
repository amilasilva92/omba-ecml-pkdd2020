import numpy as np
from collections import defaultdict


def generate_word_vecs(filename):
    # Reads the word embeddings in the format of '/data/toy/output/embs_sample.txt'
    f = open(filename, 'r')
    word_embs = defaultdict(lambda: (np.random.rand(300) - 0.5)/300)

    for line in f:
        splits = line.strip().split('\x01')
        vec = np.array([float(item.strip()) for item in splits[1][1:-1].split(',')])
        word_embs[splits[0]] = vec

    id2item = []
    id2vec = []
    item2id = {}
    for i, w in enumerate(word_embs):
        id2vec.append(word_embs[w])
        id2item.append(w)
        item2id[w] = i
    return word_embs, id2item, id2vec, item2id


class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = np.random.randn(self.hash_size, inp_dimensions)

    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return ''.join(bools.astype('str'))

    def __setitem__(self, inp_vec, label):
        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table\
            .get(hash_value, list()) + [label]

    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])


class HashTablePool:
    def __init__(self, table_size, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.table_size = table_size

        self.hash_table_pool = []
        for i in range(self.table_size):
            self.hash_table_pool.append(
                HashTable(hash_size=self.hash_size,
                          inp_dimensions=self.inp_dimensions))
        self.collisions = defaultdict(lambda: defaultdict(lambda: 0))

        self.top_rules = []
        self.min_value = 0
        self.min_id = -1
        self.num_rules = 200

    def __get_votes__(self):
        for h_table in self.hash_table_pool:
            table = h_table.hash_table
            for hash_v in table:
                for i in table[hash_v]:
                    for j in table[hash_v]:
                        if i != j:
                            self.collisions[i][j] += 1

                            if len(self.top_rules) < self.num_rules:
                                self.top_rules.append((i, j))
                                if self.min_value < self.collisions[i][j]:
                                    self.min_value = self.collisions[i][j]
                                    self.min_id = len(self.top_rules) - 1
                            else:
                                if ((i, j) in self.top_rules and
                                        self.top_rules[self.min_id] != (i, j)):
                                    continue
                                if self.min_value < self.collisions[i][j]:
                                    self.top_rules[self.min_id] = (i, j)
                                    self.min_value = self.collisions[i][j]

                                    for k, (x, y) in enumerate(self.top_rules):
                                        if self.min_value > self.collisions[x][y]:
                                            self.min_value = self.collisions[x][y]
                                            self.min_id = k
        return self.top_rules

    def _get_neighbours_(self, item, item_vec, k=10):
        item_collisions = defaultdict(lambda: 0)
        for h_table in self.hash_table_pool:
            table = h_table.hash_table

            hash_val = h_table.generate_hash(item_vec)
            for j in table[hash_val]:
                if j != item:
                    item_collisions[j] += 1

        top_nbrs = []
        for item in item_collisions:
            if len(top_nbrs) < k:
                top_nbrs.append((item, item_collisions[item]))
            else:
                min_id = 0
                min_col = np.inf

                for index, nbr in enumerate(top_nbrs):
                    if nbr[1] < min_col:
                        min_id = index
                        min_col = nbr[1]
                if min_col < item_collisions[item]:
                    top_nbrs[min_id] = (item, item_collisions[item])
        return top_nbrs

    def __setitem__(self, inp_vec, label):
        for h_table in self.hash_table_pool:
            h_table.__setitem__(inp_vec, label)

    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])


def generate_lsh_rules(id2item, id2vec, hash_table_size, hash_size):
    rand_seed = 3
    np.random.seed(rand_seed)
    pool = HashTablePool(hash_table_size, hash_size, 300)

    for label, vec in zip(id2item, id2vec):
        pool.__setitem__(vec, label)
    lsh_rules = pool.__get_votes__()

    return lsh_rules, pool


# theoratically found values
HASHTABLE_SIZE = 11
HASH_SIZE = 4

word_embs, id2item, id2vec, item2id = generate_word_vecs('../data/toy/output/embs_sample.txt')
lsh_rules, pool = generate_lsh_rules(id2item, id2vec, HASHTABLE_SIZE, HASH_SIZE)
print(lsh_rules)
