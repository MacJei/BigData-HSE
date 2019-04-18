from collections import defaultdict

import pyspark.sql.functions as f
from pyspark import SparkContext, SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, Imputer, StandardScaler, \
    OneHotEncoderEstimator, StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *

CLUSTER_MASTER = 'yarn-client'
# CLUSTER_MASTER = 'local[*]'
CLUSTER_PATH1 = 'hdfs:///data/social_task3/trainGraph'
CLUSTER_PATH2 = 'hdfs:///data/social_task3/coreDemography'
CLUSTER_PATH3 = 'hdfs:///data/social_task3/prediction.csv'
PATHS = CLUSTER_PATH1, CLUSTER_PATH2, CLUSTER_PATH3

SCHEMA1 = StructType() \
    .add('userId', IntegerType()) \
    .add('socials', MapType(IntegerType(), IntegerType()))

SCHEMA2 = StructType() \
    .add('userId', IntegerType()) \
    .add('create_date', LongType()) \
    .add('birth_date', IntegerType()) \
    .add('gender', IntegerType()) \
    .add('ID_country', LongType()) \
    .add('ID_location', IntegerType()) \
    .add('loginRegion', IntegerType())

SCHEMA3 = StructType() \
    .add('userId', IntegerType())

SCHEMAS = SCHEMA1, SCHEMA2, SCHEMA3

NAMES = '''Love, Spouse, Parent, Child, Brother/Sister, Uncle/Aunt, Relative, 
Close friend, Colleague, Schoolmate, Nephew, Grandparent, Grandchild, 
College/University fellow, Army fellow, Parent in law, Child in law, Godparent, 
Godchild, Playing together'''.split(', ')
MASK = {(1 << (i + 1)): i for i, _ in enumerate(NAMES)}
RATIO = {
    1: float(2231478) / (2231478 + 229570117),
    3: float(2043024) / (2043024 + 49505166),
    4: float(1942315) / (1942315 + 33671934),
    7: float(1662597) / (1662597 + 16014235)
}


def init():
    master, paths, schemas = CLUSTER_MASTER, PATHS, SCHEMAS

    sc = SparkContext(master)
    spark = SparkSession(sc)
    spark.catalog.clearCache()
    sql_context = SQLContext(sc)

    def parse_line(s):
        user_id, socials_str = s.split('\t')
        user_id = int(user_id)

        socials = {}
        for e in socials_str[2:-2].split('),('):
            e1, e2 = map(int, e.split(','))
            socials[e1] = e2 ^ (e2 & 1)

        return user_id, socials

    sdf = sc.textFile(paths[0]).map(parse_line)
    sdf = spark.createDataFrame(sdf, schemas[0]).persist()

    meta_df = sql_context.read.load(
        paths[1],
        format='csv',
        delimiter='\t',
        header='false',
        schema=schemas[1]
    ).persist()

    preds = sql_context.read.load(
        paths[2],
        format='csv',
        header='false',
        schema=schemas[2]
    ).persist()

    return sdf, meta_df, preds


def print_info(df):
    types = df.dtypes
    sample = df.limit(3).toPandas()
    size = df.count()
    print types
    print sample
    print 'df size', size


def make_pdf(sdf):
    return sdf.select(
        'userId',
        f.explode('socials').alias('social', 'type')
    ).persist()


def calc_common(pdf, min_socials=0):
    pdf1 = pdf.withColumnRenamed('userId', 'userId_1')
    pdf2 = pdf.withColumnRenamed('userId', 'userId_2')
    df1 = pdf1.join(pdf2, on=['social', 'type']) \
        .where(f.col('userId_1') < f.col('userId_2')) \
        .groupBy('userId_1', 'userId_2').count() \
        .withColumnRenamed('count', 'scommon') \
        .withColumn('scommon', f.col('scommon').cast('float'))

    df2 = pdf1.join(pdf2, on='social') \
        .where(f.col('userId_1') < f.col('userId_2')) \
        .groupBy('userId_1', 'userId_2').count() \
        .where(f.col('count') >= min_socials) \
        .withColumnRenamed('count', 'common') \
        .withColumn('common', f.col('common').cast('float'))

    return df2.join(df1, on=['userId_1', 'userId_2'], how='left') \
        .na.fill(0.0, subset=['scommon'])


def add_features(df, sdf, meta_df):
    def simple_feature(col):
        return f.col(col).cast('float')

    def abs_feature(col):
        return f.abs(f.col(col + '_1') - f.col(col + '_2')).cast('float')

    def same_feature(col):
        col1, col2 = f.col(col + '_1'), f.col(col + '_2')
        cond = (col1 == col2) & col1.isNotNull() & col2.isNotNull()
        return cond.cast('float')

    zdf = sdf.select('userId', f.size('socials').alias('sz'))
    df = df.join(zdf.withColumnRenamed('userId', 'userId_1'),
                 on='userId_1', how='left') \
        .withColumnRenamed('sz', 'sz_1') \
        .join(zdf.withColumnRenamed('userId', 'userId_2'),
              on='userId_2', how='left') \
        .withColumnRenamed('sz', 'sz_2') \
        .withColumn('abs_sz', abs_feature('sz'))

    meta_df1 = meta_df.toDF(*(c + '_1' for c in meta_df.columns))
    meta_df2 = meta_df.toDF(*(c + '_2' for c in meta_df.columns))
    df = df.join(meta_df1, on='userId_1', how='left') \
        .join(meta_df2, on='userId_2', how='left') \
        .withColumn('create_date_1', simple_feature('create_date_1')) \
        .withColumn('create_date_2', simple_feature('create_date_2')) \
        .withColumn('birth_date_1', simple_feature('birth_date_1')) \
        .withColumn('birth_date_2', simple_feature('birth_date_2')) \
        .withColumn('abs_create', abs_feature('create_date')) \
        .withColumn('abs_birth', abs_feature('birth_date')) \
        .withColumn('same_gender', same_feature('gender')) \
        .withColumn('same_country', same_feature('ID_country')) \
        .withColumn('same_location', same_feature('ID_location')) \
        .withColumn('same_login', same_feature('loginRegion'))

    return df


def add_label(df, pdf, ratio):
    def make_label():
        return (f.col('label') | f.col('type').isNotNull()).cast('integer')

    def make_weight():
        return f.when(f.col('label') == 1, 1 - ratio).otherwise(ratio)

    pdf1 = pdf.withColumnRenamed('userId', 'userId_1')
    pdf1 = pdf1.withColumnRenamed('social', 'userId_2')
    pdf2 = pdf.withColumnRenamed('userId', 'userId_2')
    pdf2 = pdf2.withColumnRenamed('social', 'userId_1')
    return df.join(pdf1, on=['userId_1', 'userId_2'], how='left') \
        .withColumn('label', f.col('type').isNotNull()) \
        .withColumnRenamed('type', 'type_1') \
        .join(pdf2, on=['userId_1', 'userId_2'], how='left') \
        .withColumn('label', make_label()) \
        .withColumn('weight', make_weight())


def split(df, preds):
    preds = set(i.userId for i in preds.select('userId').collect())
    cond1 = f.col('userId_1').isin(preds) | f.col('userId_2').isin(preds)
    cond2 = (f.col('userId_1') % 11 == 7) | (f.col('userId_2') % 11 == 7)
    cond3 = f.col('label') == 0
    train_cond = ~(cond2 & cond3)
    train_df = df.where(train_cond)
    test_cond = cond1 & cond2 & cond3
    test_df = df.where(test_cond)
    return train_df, test_df


def make_pipeline():
    cols_d = ['common', 'abs_sz',  # 'scommon'
              'create_date_1', 'create_date_2', 'birth_date_1', 'birth_date_2',
              'abs_create', 'abs_birth']
    cols_zo = ['same_gender', 'same_country', 'same_location', 'same_login']
    cols_c = ['gender_1']  # 'ID_country_1', 'ID_location_1'

    stages = []

    # Numerical
    imputer = Imputer(inputCols=cols_d, outputCols=cols_d)
    assembler = VectorAssembler(
        inputCols=cols_d + cols_zo,
        outputCol='raw_num_features'
    )
    scaler = StandardScaler(
        withMean=True, inputCol='raw_num_features', outputCol='num_features'
    )
    stages += [imputer, assembler, scaler]

    # Categorical
    for col in cols_c:
        indexer = StringIndexer(
            inputCol=col, outputCol=col + '_index', handleInvalid='keep'
        )
        encoder = OneHotEncoderEstimator(
            inputCols=[indexer.getOutputCol()],
            outputCols=[col + '_vec'], handleInvalid='keep'
        )
        stages += [indexer, encoder]

    if len(cols_c):
        assembler = VectorAssembler(
            inputCols=[c + '_vec' for c in cols_c],
            outputCol='cat_features'
        )
        stages += [assembler]

    # Combine
    assembler = VectorAssembler(
        inputCols=['num_features', 'cat_features'],
        outputCol='features'
    )
    stages += [assembler]

    pipeline = Pipeline(stages=stages)
    return pipeline


def compose_pred(pred_df, preds):
    second_f = udf(lambda v: float(v[1]), FloatType())
    pred_df = pred_df.where(f.col('prediction').cast('integer') == 1) \
        .select('userId_1', 'userId_2',
                second_f('probability').alias('prob'))
    ldf = pred_df.withColumnRenamed('userId_1', 'userId') \
        .withColumnRenamed('userId_2', 'social') \
        .groupBy('userId') \
        .agg(f.collect_list(f.struct('prob', 'social')).alias('lsocials'))
    rdf = pred_df.withColumnRenamed('userId_2', 'userId') \
        .withColumnRenamed('userId_1', 'social') \
        .groupBy('userId') \
        .agg(f.collect_list(f.struct('prob', 'social')).alias('rsocials'))

    def compose_f(ls, rs):
        ans = []
        for elem in (ls or []):
            ans.append((elem['prob'], elem['social']))
        for elem in (rs or []):
            ans.append((elem['prob'], elem['social']))
        return ans

    schema = ArrayType(StructType([
        StructField('prob', FloatType(), False),
        StructField('id', IntegerType(), False)
    ]))
    compose_f = udf(compose_f, schema)
    pred_df = preds.join(ldf, on='userId', how='left') \
        .join(rdf, on='userId', how='left') \
        .withColumn('list', compose_f('lsocials', 'rsocials')) \
        .select('userId', 'list')

    # Take 1000000 most probable
    pred_df = pred_df.toPandas()
    lls = []
    for _, row in pred_df.iterrows():
        for e in row['list']:
            lls.append((e['prob'], e['id'], row['userId']))
    lls.sort(reverse=True)
    lls = lls[:1000000]
    d = defaultdict(list)
    for prob, id_, userId in lls:
        d[userId].append(id_)
    lists = [' '.join(str(e) for e in d[user_id])
             for user_id in pred_df.userId]
    pred_df['list'] = lists

    return pred_df


def main():
    sdf, meta_df, preds = init()

    pdf = make_pdf(sdf)
    min_socials = 3
    assert min_socials in RATIO, 'Should precalculate ratio first!'
    df = calc_common(pdf, min_socials=min_socials)

    df = add_features(df, sdf, meta_df)
    df = add_label(df, pdf, ratio=RATIO[min_socials])

    train_df, test_df = split(df, preds)

    pipeline = make_pipeline()
    pipeline_model = pipeline.fit(train_df)
    train_df = pipeline_model.transform(train_df)
    lr = LogisticRegression(maxIter=100, weightCol='weight',
                            regParam=0.0, threshold=0.5)
    model = lr.fit(train_df)

    test_df = pipeline_model.transform(test_df)
    pred_df = model.transform(test_df)

    pred_df = compose_pred(pred_df, preds)
    pred_df.to_csv('pred.csv', header=False, index=False, quoting=2)


if __name__ == '__main__':
    main()
