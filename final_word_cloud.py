import sys
from collections import Counter

import matplotlib.pyplot as plt
from pyspark.ml.feature import HashingTF, StopWordsRemover
from sparknlp.annotator import *
from sparknlp.base import *
from wordcloud import WordCloud

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

import sys

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
from pyspark.ml.pipeline import Pipeline
# from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.sql.functions import udf

from pyspark.sql import SparkSession, types, functions
from pyspark.sql.functions import collect_list

KEY_SPACE = 'bigp18'
TABLE_REVIEW = 'review'

cluster_seeds = ['199.60.17.188', '199.60.17.216', '127.0.0.1']
spark = SparkSession.builder \
    .appName("ner") \
    .master("local[*]") \
    .config('spark.cassandra.connection.host', ','.join(cluster_seeds)) \
    .config("spark.driver.memory", "4G") \
    .config("spark.driver.maxResultSize", "2G") \
    .config("spark.driver.extraClassPath", "lib/spark-nlp-assembly-1.7.3.jar") \
    .config("spark.kryoserializer.buffer.max", "500m") \
    .getOrCreate()

assert spark.version >= '2.3'  # make sure we have Spark 2.3+


def get_attributes(attributes):
    list_of_tokens = ''
    for attribute in attributes:
        list_of_tokens = (list_of_tokens + "," + attribute[3])
    return list_of_tokens


def main(business_id_arg):
    concat_list = udf(lambda lst: ", ".join(lst), types.StringType())

    reviews_df = spark.read.format("org.apache.spark.sql.cassandra") \
        .options(table=TABLE_REVIEW, keyspace=KEY_SPACE) \
        .load()

    review_filter = reviews_df.filter(reviews_df.business_id == business_id_arg)
    review_concatenate = review_filter.groupby('business_id').agg(collect_list('review').alias("review"))
    review_concatenate.show()
    train_fin = review_concatenate.withColumn("review", concat_list("review"))
    train_fin = train_fin.withColumn("review", functions.regexp_replace(train_fin.review, "[^0-9A-Za-z ,]", ""))

    # Create a new pipeline to create Tokenizer and Lemmatizer
    documentAssembler = DocumentAssembler().setInputCol("review").setOutputCol("document")
    tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
    lemmatizer = Lemmatizer().setInputCols(["token"]).setOutputCol("lemma") \
        .setDictionary("lemmas001.txt", key_delimiter=" ", value_delimiter="\t")

    pipeline = Pipeline(stages=[documentAssembler, tokenizer, lemmatizer])
    pipelineFit = pipeline.fit(train_fin)

    train_df = pipelineFit.transform(train_fin)
    train_df.select('lemma').show(truncate=False)
    price_range_udf = functions.UserDefinedFunction(lambda attributes: get_attributes(attributes), types.StringType())
    train_df = train_df.withColumn('lemma', price_range_udf(train_df['lemma']))
    train_df = train_df.withColumn('lemma', functions.split(train_df['lemma'], ",").cast('array<string>'))

    # Create a new pipeline to remove the stop words
    test_review = train_df.select("lemma")
    stop_words_remover = StopWordsRemover(inputCol="lemma", outputCol="filtered")
    hash_tf = HashingTF(numFeatures=2 ** 16, inputCol="lemma", outputCol='tf')
    pipeline_too_remove_stop_words = Pipeline(stages=[hash_tf, stop_words_remover])
    pipeline_fit = pipeline_too_remove_stop_words.fit(train_df)
    test_df = pipeline_fit.transform(test_review)
    test_df.show()

    token_array = test_df.select('filtered').rdd.flatMap(lambda row: row).collect()

    counts = Counter(token_array[0])
    word_cloud = WordCloud(
        background_color='white',
        max_words=100,
        max_font_size=50,
        min_font_size=10,
        random_state=40
    ).fit_words(counts)

    plt.imshow(word_cloud)
    plt.axis('off')  # remove axis
    plt.show()


if __name__ == '__main__':
    business_id = sys.argv[1]
    # output = sys.argv[2]
    main(business_id)
