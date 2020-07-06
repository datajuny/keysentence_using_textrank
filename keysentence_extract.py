# -*- coding: utf-8 -*-
from konlpy.tag import Komoran
from nltk.tokenize import sent_tokenize
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from collections import OrderedDict
from sklearn.preprocessing import normalize

import re
import nltk
import numpy as np
import time
import math
import scipy as sp

# Not_Key_sentence_RESUME 테이블에 insert
def notkeysen_insert(ModelID, URI, RESUME_NO, SENTENCE, RESUME):

    try:
        conn = pymysql.connect(host=config.DATABASE_CONFIG['host'], port=config.DATABASE_CONFIG['port'],
                               user=config.DATABASE_CONFIG['user'],
                               passwd=config.DATABASE_CONFIG['password'],
                               db=config.DATABASE_CONFIG['dbname'],
                               charset=config.DATABASE_CONFIG['charset'])
        curs = conn.cursor()

        SQL_Keyword_Insert = "INSERT INTO `RESULT_PFA02_NOTKEYSEN` \
                        (`ModelID`, `URI`, `RESUME_NO`, `SENTENCE`, `RESUME`) \
                          VALUES ('%s', '%s', '%s', '%s', '%s')" % (ModelID, URI, RESUME_NO, SENTENCE, RESUME)

        curs.execute(SQL_Keyword_Insert)
        curs.fetchall()

        conn.commit()
        conn.close()

        # print("URI : ", URI, " / RESUME_NO : ", RESUME_NO, " => [RESULT_PFA06_KEYWORD] 테이블 입력 완료!")

    except Exception as e:
        print('RESULT_PFA02_NotKeySen TABLE Insert EXCEPTION ', str(e))
        print("해당 URI의 자소서는 Insert할 수 없습니다. ==>", URI, "자소서 번호 ==> ", RESUME_NO)
        pass


def notkeysen_insert(ModelID, URI, RESUME_NO, SENTENCE, RESUME):

    try:
        conn = pymysql.connect(host=config.DATABASE_CONFIG['host'], port=config.DATABASE_CONFIG['port'],
                               user=config.DATABASE_CONFIG['user'],
                               passwd=config.DATABASE_CONFIG['password'],
                               db=config.DATABASE_CONFIG['dbname'],
                               charset=config.DATABASE_CONFIG['charset'])
        curs = conn.cursor()

        SQL_Keyword_Insert = "INSERT INTO `RESULT_PFA02_NOTKEYSEN` \
                        (`ModelID`, `URI`, `RESUME_NO`, `SENTENCE`, `RESUME`) \
                          VALUES ('%s', '%s', '%s', '%s', '%s')" % (ModelID, URI, RESUME_NO, SENTENCE, RESUME)

        curs.execute(SQL_Keyword_Insert)
        curs.fetchall()

        conn.commit()
        conn.close()

        # print("URI : ", URI, " / RESUME_NO : ", RESUME_NO, " => [RESULT_PFA06_KEYWORD] 테이블 입력 완료!")

    except Exception as e:
        print('RESULT_PFA02_NotKeySen TABLE Insert EXCEPTION ', str(e))
        print("해당 URI의 자소서는 Insert할 수 없습니다. ==>", URI, "자소서 번호 ==> ", RESUME_NO)
        pass


class KeysentenceSummarizer:

    def __init__(self, sents=None, tokenize=None, min_count=2,
                 min_sim=0.3, similarity=None, vocab_to_idx=None,
                 df=0.85, max_iter=30, verbose=False):

        self.tokenize = tokenize
        self.min_count = min_count
        self.min_sim = min_sim
        self.similarity = similarity
        self.vocab_to_idx = vocab_to_idx
        self.df = df
        self.max_iter = max_iter
        self.verbose = verbose

        if sents is not None:
            self.train_textrank(sents)

    def train_textrank(self, sents, bias=None):

        g = sent_graph(sents, self.tokenize, self.min_count,
                       self.min_sim, self.similarity, self.vocab_to_idx, self.verbose)
        self.R = pagerank(g, self.df, self.max_iter, bias).reshape(-1)
        if self.verbose:
            print('trained TextRank. n sentences = {}'.format(self.R.shape[0]))

    def summarize(self, sents, topk=30, bias=None):

        n_sents = len(sents)
        if isinstance(bias, np.ndarray):
            if bias.shape != (n_sents,):
                raise ValueError('The shape of bias must be (n_sents,) but {}'.format(bias.shape))
        elif bias is not None:
            raise ValueError('The type of bias must be None or numpy.ndarray but the type is {}'.format(type(bias)))

        self.train_textrank(sents, bias)
        idxs = self.R.argsort()[-topk:]
        keysents = [(idx, self.R[idx], sents[idx]) for idx in reversed(idxs)]
        return keysents


def pagerank(x, df=0.85, max_iter=30, bias=None):
    assert 0 < df < 1

    # initialize
    A = normalize(x, axis=0, norm='l1')
    R = np.ones(A.shape[0]).reshape(-1, 1)

    # check bias
    if bias is None:
        bias = (1 - df) * np.ones(A.shape[0]).reshape(-1, 1)
    else:
        bias = bias.reshape(-1, 1)
        bias = A.shape[0] * bias / bias.sum()
        assert bias.shape[0] == A.shape[0]
        bias = (1 - df) * bias

    # iteration
    for _ in range(max_iter):
        R = df * (A * R) + bias

    return R


def sent_graph(sents, tokenize=None, min_count=2, min_sim=0.3,
               similarity=None, vocab_to_idx=None, verbose=False):
    if vocab_to_idx is None:
        idx_to_vocab, vocab_to_idx = scan_vocabulary(sents, tokenize, min_count)
    else:
        idx_to_vocab = [vocab for vocab, _ in sorted(vocab_to_idx.items(), key=lambda x: x[1])]

    x = vectorize_sents(sents, tokenize, vocab_to_idx)
    if similarity == 'cosine':
        x = numpy_cosine_similarity_matrix(x, min_sim, verbose, batch_size=1000)
    else:
        x = numpy_textrank_similarity_matrix(x, min_sim, verbose, batch_size=1000)
    return x


def vectorize_sents(sents, tokenize, vocab_to_idx):
    rows, cols, data = [], [], []
    for i, sent in enumerate(sents):
        counter = Counter(tokenize(sent))
        for token, count in counter.items():
            j = vocab_to_idx.get(token, -1)
            if j == -1:
                continue
            rows.append(i)
            cols.append(j)
            data.append(count)
    n_rows = len(sents)
    n_cols = len(vocab_to_idx)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))


def numpy_cosine_similarity_matrix(x, min_sim=0.3, verbose=True, batch_size=1000):
    n_rows = x.shape[0]
    mat = []
    for bidx in range(math.ceil(n_rows / batch_size)):
        b = int(bidx * batch_size)
        e = min(n_rows, int((bidx + 1) * batch_size))
        psim = 1 - pairwise_distances(x[b:e], x, metric='cosine')
        rows, cols = np.where(psim >= min_sim)
        data = psim[rows, cols]
        mat.append(csr_matrix((data, (rows, cols)), shape=(e - b, n_rows)))
        if verbose:
            print('\rcalculating cosine sentence similarity {} / {}'.format(b, n_rows), end='')
    mat = sp.sparse.vstack(mat)
    if verbose:
        print('\rcalculating cosine sentence similarity was done with {} sents'.format(n_rows))
    return mat


def numpy_textrank_similarity_matrix(x, min_sim=0.3, verbose=True, min_length=1, batch_size=1000):
    n_rows, n_cols = x.shape

    # Boolean matrix
    rows, cols = x.nonzero()
    data = np.ones(rows.shape[0])
    z = csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    # Inverse sentence length
    size = np.asarray(x.sum(axis=1)).reshape(-1)
    size[np.where(size <= min_length)] = 10000
    size = np.log(size)

    mat = []
    for bidx in range(math.ceil(n_rows / batch_size)):

        # slicing
        b = int(bidx * batch_size)
        e = min(n_rows, int((bidx + 1) * batch_size))

        # dot product
        inner = z[b:e, :] * z.transpose()

        # sentence len[i,j] = size[i] + size[j]
        norm = size[b:e].reshape(-1, 1) + size.reshape(1, -1)
        norm = norm ** (-1)
        norm[np.where(norm == np.inf)] = 0

        # normalize
        sim = inner.multiply(norm).tocsr()
        rows, cols = (sim >= min_sim).nonzero()
        data = np.asarray(sim[rows, cols]).reshape(-1)

        # append
        mat.append(csr_matrix((data, (rows, cols)), shape=(e - b, n_rows)))

        if verbose:
            print('\rcalculating textrank sentence similarity {} / {}'.format(b, n_rows), end='')

    mat = sp.sparse.vstack(mat)
    if verbose:
        print('\rcalculating textrank sentence similarity was done with {} sents'.format(n_rows))

    return mat


def graph_with_python_sim(tokens, verbose, similarity, min_sim):
    if similarity == 'cosine':
        similarity = cosine_sent_sim
    elif callable(similarity):
        similarity = similarity
    else:
        similarity = textrank_sent_sim

    rows, cols, data = [], [], []
    n_sents = len(tokens)
    for i, tokens_i in enumerate(tokens):
        if verbose and i % 1000 == 0:
            print('\rconstructing sentence graph {} / {} ...'.format(i, n_sents), end='')
        for j, tokens_j in enumerate(tokens):
            if i >= j:
                continue
            sim = similarity(tokens_i, tokens_j)
            if sim < min_sim:
                continue
            rows.append(i)
            cols.append(j)
            data.append(sim)
    if verbose:
        print('\rconstructing sentence graph was constructed from {} sents'.format(n_sents))
    return csr_matrix((data, (rows, cols)), shape=(n_sents, n_sents))


def textrank_sent_sim(s1, s2):
    n1 = len(s1)
    n2 = len(s2)
    if (n1 <= 1) or (n2 <= 1):
        return 0
    common = len(set(s1).intersection(set(s2)))
    base = math.log(n1) + math.log(n2)
    return common / base


def cosine_sent_sim(s1, s2):
    if (not s1) or (not s2):
        return 0

    s1 = Counter(s1)
    s2 = Counter(s2)
    norm1 = math.sqrt(sum(v ** 2 for v in s1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in s2.values()))
    prod = 0
    for k, v in s1.items():
        prod += v * s2.get(k, 0)
    return prod / (norm1 * norm2)


def scan_vocabulary(sents, tokenize=None, min_count=2):
    counter = Counter(w for sent in sents for w in tokenize(sent))
    counter = {w: c for w, c in counter.items() if c >= min_count}
    idx_to_vocab = [w for w, _ in sorted(counter.items(), key=lambda x: -x[1])]
    vocab_to_idx = {vocab: idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx


def tokenize_sents(sents, tokenize):
    return [tokenize(sent) for sent in sents]


def dict_to_mat(d, n_rows, n_cols):
    rows, cols, data = [], [], []
    for (i, j), v in d.items():
        rows.append(i)
        cols.append(j)
        data.append(v)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))


def komoran_tokenizer(sent):
    words = komoran.pos(sent, join=True)
    words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]

    return words


komoran = Komoran()

def keysentence_extract(RESUME):

    try:
    
        RESUME = re.sub("[^ㄱ-ㅣ가-힣|a-zA-Z|0-9|.]+", " ", RESUME) 

        RESUME_SENT = sent_tokenize(RESUME)

        summarizer = KeysentenceSummarizer(tokenize=komoran_tokenizer, min_sim=0.0000001)
        keysents = summarizer.summarize(RESUME_SENT, topk=100)


        # 변수 선언
        sent_idx_ordered = OrderedDict()  # textrank 계산이 완료된 자소서를 담는다.

        NotKeySen_YN = "N"  # NoKeySen이 있는 자소서인지 표기한다.

        recover_resume = ""  # 해당 문장에 태그를 붙여 자소서 원문을 만든다.
        NotKeySen = ""  # 비중요 문장을 담는다.

        # 계산 결과를 담는다.
        for sent_idx, score, sent in keysents:
            sent_idx_ordered[sent_idx] = [score, sent]
        
    except Exception as e:
        return "중요문장이 추출되지 않았습니다."
    
    return sent_idx_ordered[0][1]
