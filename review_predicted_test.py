import sys
import datetime
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('tmax model tester').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegressionModel

review_schema = types.StructType([
    types.StructField('review_id', types.StringType()),
    types.StructField('business_id', types.StringType()),
    types.StructField('zip_code', types.IntegerType()),
    types.StructField('user_id', types.StringType()),
    types.StructField('review', types.StringType()),
    types.StructField('r_stars', types.IntegerType()),
    types.StructField('r_date', types.DateType())
])


def test_model(model_file):

    # load the model
    model = LogisticRegressionModel.load(model_file)
    #taken the value of the temperature from the internet
    #inputs = spark.read.csv(inputs, schema = review_schema)
    inputs = [
    ('abcd', 'asff',123,'asasf','I love it',5,datetime.date(2018, 11, 13)),
    ('abcdgfgdf', 'asffasdas',123,'asasf','I hate it',1,datetime.date(2018, 11, 13))
             ]

    test = spark.createDataFrame(inputs,review_schema)
    test_review = test.select("review")
    predictions = model.transform(test_review)
    predictions_ren = predictions.selectExpr("review as review1")
    test = test.withColumn('row_index', functions.monotonically_increasing_id())
    predictions_ren = predictions_ren.withColumn('row_index', functions.monotonically_increasing_id())
    test = test.join(predictions_ren, on=["row_index"]).sort("row_index").drop("row_index")
    test = test.drop("review1")
    test.show(10)
    test.write.csv('filtered.csv')

if __name__ == '__main__':
    model_file = sys.argv[1]
    #inputs = sys.argv[2]
    test_model(model_file)
