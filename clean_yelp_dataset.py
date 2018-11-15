import sys

from cassandra.cluster import Cluster
from pyspark.sql import SparkSession, types

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

cluster_seeds = ['192.168.0.1', '192.168.0.2']
spark = SparkSession.builder.appName('CleanYelpDataset').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3'  # make sure we have Spark 2.3+

review_schema = types.StructType([
    types.StructField('review_id', types.StringType(), True),
    types.StructField('business_id', types.StringType(), True),
    types.StructField('user_id', types.StringType(), True),
    types.StructField('stars', types.IntegerType(), True),
    types.StructField('text', types.StringType(), True),
    types.StructField('date', types.DateType(), True)
])

business_schema = types.StructType([
    types.StructField('business_id', types.StringType(), True),
    types.StructField('city', types.StringType(), True),
    types.StructField('state', types.StringType(), True),
    types.StructField('stars', types.DoubleType(), True),
    types.StructField('review_count', types.LongType(), True),
    types.StructField('categories', types.StringType(), True),
    types.StructField('postal_code', types.StringType(), True),
    types.StructField('latitude', types.DoubleType(), True),
    types.StructField('longitude', types.DoubleType(), True),
    types.StructField('attributes', types.StructType(
        [types.StructField("RestaurantsPriceRange2", types.StringType(), True)]
    ), False),
    types.StructField('is_open', types.IntegerType(), True),
    types.StructField('name', types.StringType(), True)
])

user_schema = types.StructType([
    types.StructField('user_id', types.StringType(), True),
    types.StructField('average_stars', types.DoubleType(), True),
    types.StructField('review_count', types.LongType(), True),
    types.StructField('yelping_since', types.DateType(), True)
])

business_states = ['ON', 'QC', 'NS', 'NB', 'MB', 'BC', 'PE', 'SK', 'AB', 'NL']
key_space = 'BigP18'


def parse_json_files_to_df(businesses_json_file_path_arg, review_json_file_path_arg, user_json_file_path_arg):
    # cluster = Cluster()
    # session = cluster.connect('')
    # return

    # Read business JSON file using the schema
    business_df = spark.read.json(businesses_json_file_path_arg, schema=business_schema)
    # Filter all the businesses which are still open in the business_states
    business_df = business_df.filter((business_df['is_open'] == 1) &
                                     business_df['state'].isin(business_states))
    # businesses_df.write.format("org.apache.spark.sql.cassandra").options(table='business', keyspace=key_space).save()

    # Read review JSON file using the schema
    review_df = spark.read.json(review_json_file_path_arg, schema=review_schema)
    review_df = review_df.join(business_df, business_df['business_id'] == review_df['business_id'])
    # reviews_df.write.format("org.apache.spark.sql.cassandra").options(table='review', keyspace=key_space).save()

    # Read user JSON file using the schema
    users_df = spark.read.json(user_json_file_path_arg, schema=user_schema)
    # users_df.write.format("org.apache.spark.sql.cassandra").options(table='users', keyspace=key_space).save()

    # businesses_df.show()
    # reviews_df.show()
    # users_df.show()
    # weather_etl_data.write.json(output_arg, compression='gzip', mode='overwrite')


# Input will be the path of the business, review and user JSON file
if __name__ == '__main__':
    businesses_json_file_path = sys.argv[1]
    review_json_file_path = sys.argv[2]
    user_json_file_path = sys.argv[3]
    parse_json_files_to_df(businesses_json_file_path, review_json_file_path, user_json_file_path)
