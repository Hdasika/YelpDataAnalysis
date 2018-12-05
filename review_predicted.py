import sys
import datetime
from pyspark.ml.classification import LogisticRegressionModel
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('tmax model tester').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors

review_schema = types.StructType([
    types.StructField('review', types.StringType())
])


def test_model(model_file):

    # load the model
    #model = PipelineModel.load(model_file)
    model = LogisticRegressionModel.load(model_file)
    #taken the value of the temperature from the internet
    #inputs = spark.read.csv(inputs, schema = review_schema)
    inputs_1 = [["I love the restaurant"]]
    test = spark.createDataFrame(inputs_1,review_schema)
    test.show()
    predictions = model.transform(test)
    predictions.show()
    #print(predictions.select('prediction').collect())

if __name__ == '__main__':
    model_file = sys.argv[1]
    #inputs = sys.argv[2]
    test_model(model_file)
