# Following changes are required: 1. The dataset is too small, synthetic data is suggested but not a good practice. 2. I am getting evaluator score to be 100%, something is fishy 3. I want to try polling/voting method if time permits 4. A  file like weather-tomorrow in the assignment is pending.
import sys
import re
from datetime import date, datetime

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import *
from pyspark.ml.classification import LinearSVC
from pyspark.sql import SparkSession, functions, types
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC


assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

from pyspark.sql import SparkSession, types
from pyspark import Row

spark = SparkSession.builder.appName('ML Pipeline').getOrCreate()
assert spark.version >= '2.3'  # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')


def construct_metadata_rdd(input_data: str):
   data = input_data.split()
   yield Row(int(data[0]), int(data[1]), float(data[2]), int(data[3]), datetime.strptime(data[4], '%Y-%m-%d'))

def construct_review_rdd(input_data: str):
   data = input_data.split()
   yield Row(int(data[0]), str(data[1]),datetime.strptime(data[2], '%Y-%m-%d') , str(data[3:]))

def main(model_file):
   review_schema = types.StructType([
       types.StructField('user_id', types.IntegerType(), True),
       types.StructField('name', types.StringType(), True),
       types.StructField('date', types.DateType(), True),
       types.StructField('review', types.StringType(), True)
   ])

   metadata_schema = types.StructType([
       types.StructField('user_id', types.IntegerType(), True),
       types.StructField('prod_id', types.IntegerType(), True),
       types.StructField('rating', types.FloatType(), True),
       types.StructField('label', types.IntegerType(), True),
       types.StructField('date', types.DateType(), True)
   ])

   metadata_text_file = spark.sparkContext.textFile('metadata')
   metadata = spark.createDataFrame(metadata_text_file.flatMap(construct_metadata_rdd), schema=metadata_schema)
   review_content_text_file = spark.sparkContext.textFile('reviewcontent')
   reviewcontent = spark.createDataFrame(review_content_text_file.flatMap(construct_review_rdd), schema=review_schema)
   cond = [reviewcontent.date == metadata.date, reviewcontent.user_id == metadata.user_id]
   combine = reviewcontent.join(metadata, cond)
   train = combine.select(combine.label, combine.review)
   train_fin = train.withColumn('label', regexp_replace('label', '-1', '0'))
   train_fin_1 = train_fin.withColumn("label", train_fin["label"].cast(DoubleType()))
   #train.show(10)
   #train_neg = train.filter(train.label == -1)
   #train_1 = train.union(train_neg)
   #train_2 = train_1.union(train_neg)
   #train_balanced = train_2.union(train_neg)
   #train_balanced.show(10)
   (train_set, val_set, test_set) = train_fin_1.randomSplit([0.98, 0.01, 0.01], seed = 100)
   tokenizer = Tokenizer(inputCol="review", outputCol="words")
   hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
   idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5)
   label_stringIdx = StringIndexer(inputCol = "label", outputCol = "target",handleInvalid="error")
   pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])

   pipelineFit = pipeline.fit(train_set)
   train_df = pipelineFit.transform(train_set)
   val_df = pipelineFit.transform(val_set)


   lr = LogisticRegression(maxIter=100)
   lrModel = lr.fit(train_df)
   predictions = lrModel.transform(val_df)
   lrModel.write().overwrite().save(model_file)




if __name__ == '__main__':
   model_file = sys.argv[1]
   main(model_file)