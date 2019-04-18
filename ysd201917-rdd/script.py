import math
import re

from pyspark import SparkContext

MASTER = 'yarn-client'
# MASTER = 'local[2]'
PATH = 'hdfs:///data/wiki/en_articles_part/articles-part'
STOP_WORDS_PATH = 'hdfs:///data/wiki/stop_words_en-xpo6.txt'


def collect_sw(sc):
    stop_words = sc.textFile(STOP_WORDS_PATH) \
        .map(lambda x: x.strip().lower()) \
        .collect()
    return set(stop_words)


def parse_article(line, stop_words):
    _, text = unicode(line.rstrip()).split('\t', 1)
    text = re.sub(r'^\W+|\W+$', '', text, flags=re.UNICODE)
    words = re.split(r'\W*\s+\W*', text, flags=re.UNICODE)
    words = [word.lower() for word in words]
    words = [word for word in words if word not in stop_words]
    return words


def npmi(pab, pa, pb):
    pmi = math.log(pab / (pa * pb))
    return pmi / (-math.log(pab))


def main():
    sc = SparkContext(MASTER)

    stop_words = collect_sw(sc)
    words = sc.textFile(PATH) \
        .map(lambda x: parse_article(x, stop_words))

    words_c = words \
        .flatMap(lambda x: x) \
        .map(lambda x: (x, 1)) \
        .reduceByKey(lambda a, b: a + b)
    total_words = words_c \
        .reduce(lambda a, b: (None, a[1] + b[1]))[1]
    pa = words_c \
        .map(lambda p: (p[0], float(p[1]) / total_words)) \
        .collectAsMap()

    pairs_c = words \
        .flatMap(lambda x: zip(x[:-1], x[1:])) \
        .map(lambda x: (x, 1)) \
        .reduceByKey(lambda a, b: a + b)
    total_pairs = pairs_c \
        .reduce(lambda a, b: (None, a[1] + b[1]))[1]
    pab = pairs_c \
        .filter(lambda p: p[1] >= 500) \
        .map(lambda p: (p[0], float(p[1]) / total_pairs))

    ans = pab \
        .map(lambda p: (p[0], (p[1], pa[p[0][0]], pa[p[0][1]]))) \
        .map(lambda p: (p[0], npmi(*p[1]))) \
        .map(lambda p: (u'{}_{}'.format(*p[0]), p[1])) \
        .top(39, key=lambda p: p[1])

    for k, _ in ans:
        print u'{}'.format(k)


if __name__ == '__main__':
    main()
