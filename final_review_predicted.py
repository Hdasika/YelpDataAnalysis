import sys

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import SparkSession, functions

# spark = SparkSession.builder.appName('tmax model tester').getOrCreate()
cluster_seeds = ['199.60.17.188', '199.60.17.216', '127.0.0.1']
spark = SparkSession.builder \
    .config('spark.cassandra.connection.host', ','.join(cluster_seeds)) \
    .appName('ModelTester') \
    .getOrCreate()
assert spark.version >= '2.3'  # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')
from pyspark.ml.classification import LogisticRegressionModel

KEY_SPACE = 'bigp18'
TABLE_REVIEW = 'review'

ROOT_PATH = '/Users/Salil/bigdata/projects/YelpDataAnalysis/model_nlp'  # load your model


def test_model():
    # load the model
    model = LogisticRegressionModel.load(ROOT_PATH)

    reviews_df = spark.read.format("org.apache.spark.sql.cassandra") \
        .options(table=TABLE_REVIEW, keyspace=KEY_SPACE) \
        .load()
    reviews_df = reviews_df.withColumn('review', functions.regexp_replace(reviews_df.review, "[^0-9A-Za-z ,]", ""))

    test_review = reviews_df.select("review")

    tokenizer = Tokenizer(inputCol="review", outputCol="words")
    hashtf = HashingTF(numFeatures=2 ** 16, inputCol="words", outputCol='tf')
    idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5)

    pipeline = Pipeline(stages=[tokenizer, hashtf, idf])

    pipelineFit = pipeline.fit(test_review)
    test_df = pipelineFit.transform(test_review)
    predictions = model.transform(test_df)

    prediction_sel = predictions.select("review", "prediction")

    predictions_ren = prediction_sel.selectExpr("review as review1", "prediction")
    test = reviews_df.withColumn('row_index', functions.monotonically_increasing_id())
    predictions_ren = predictions_ren.withColumn('row_index', functions.monotonically_increasing_id())
    test = test.join(predictions_ren, on=["row_index"]).sort("row_index").drop("row_index")
    test = test.drop("review1")

    test.write \
        .mode('overwrite') \
        .option("multiLine", "true") \
        .csv(path=TABLE_REVIEW, header=True)


if __name__ == '__main__':
    test_model()
