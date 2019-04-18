import pyspark.sql.functions as f
from pyspark import SparkContext, SQLContext

LOCAL_MASTER = 'local[2]'
LOCAL_PATH = 'lsml_hw2_samples/graphDFSampleQuater'
CLUSTER_MASTER = 'yarn-client'
CLUSTER_PATH = 'hdfs:///data/graphDFQuarter'
CLUSTER_SAMPLE_PATH = 'hdfs:///data/graphDFSampleQuater'
LOCAL = False
SAMPLE = False


def init(local=False, sample=False):
    if local:
        master, path = LOCAL_MASTER, LOCAL_PATH
    else:
        master = CLUSTER_MASTER
        if sample:
            path = CLUSTER_SAMPLE_PATH
        else:
            path = CLUSTER_PATH

    sc = SparkContext(master)
    sql_context = SQLContext(sc)
    df = sql_context.read.parquet(path)

    return sc, df


def main():
    sc, df = init(local=LOCAL, sample=SAMPLE)

    pdf = df.select('user', f.explode('friends').alias('friend'))
    pdf1 = pdf.withColumnRenamed('user', 'user_1')
    pdf2 = pdf.withColumnRenamed('user', 'user_2')
    ans = pdf1.join(pdf2, on='friend') \
        .where(f.col('user_1') < f.col('user_2')) \
        .groupBy('user_1', 'user_2').count() \
        .orderBy(f.col('count').desc(),
                 f.col('user_1').desc(),
                 f.col('user_2').desc()) \
        .limit(49) \
        .toPandas()

    for _, row in ans.iterrows():
        print row['count'], row.user_1, row.user_2


if __name__ == '__main__':
    main()
