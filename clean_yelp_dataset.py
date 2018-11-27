import sys

from cassandra.cluster import Cluster
from pyspark.sql import SparkSession, types, functions

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

cluster_seeds = ['199.60.17.188', '199.60.17.216']
spark = SparkSession.builder \
    .config('spark.cassandra.connection.host', ','.join(cluster_seeds)) \
    .appName('CleanYelpDataset') \
    .getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3'  # make sure we have Spark 2.3+

business_states = ['PA', 'NV', 'NC', 'IL', 'OH', 'AZ']

KEY_SPACE = 'bigp18'
TABLE_BUSINESS = 'business'
TABLE_USER = 'user'
TABLE_REVIEW = 'review'

QUERY_CREATE_KEY_SPACE = "CREATE KEYSPACE IF NOT EXISTS " + KEY_SPACE + \
                         " WITH replication = { 'class' : 'SimpleStrategy', 'replication_factor' : 3 };"

QUERY_CREATE_BUSINESS_TABLE = "CREATE TABLE IF NOT EXISTS " + TABLE_BUSINESS + " ( b_id TEXT PRIMARY KEY, " \
                                                                               "city TEXT, " \
                                                                               "state TEXT, " \
                                                                               "review_count INT, " \
                                                                               "categories TEXT, " \
                                                                               "postal_code TEXT, " \
                                                                               "latitude DOUBLE, " \
                                                                               "longitude DOUBLE, " \
                                                                               "pricerange INT, " \
                                                                               "b_stars FLOAT );"
QUERY_CREATE_USER_TABLE = "CREATE TABLE IF NOT EXISTS " + TABLE_USER + " ( user_id TEXT PRIMARY KEY, " \
                                                                       "review_count INT, " \
                                                                       "yelping_since DATE, " \
                                                                       "average_stars FLOAT );"
QUERY_CREATE_REVIEW_TABLE = "CREATE TABLE IF NOT EXISTS " + TABLE_REVIEW + " ( review_id TEXT PRIMARY KEY, " \
                                                                           "business_id TEXT, " \
                                                                           "user_id TEXT, " \
                                                                           "review TEXT, " \
                                                                           "r_stars INT, " \
                                                                           "r_date DATE );"


def get_price_range(attributes):
    if attributes is not None and attributes['RestaurantsPriceRange2'] is not None:
        price_range = int(attributes['RestaurantsPriceRange2'])
    else:
        price_range = 0
    return price_range


# Read and store business JSON file using the schema
def process_business_json(input_json_business):
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
    # Read business JSON file using the schema
    business_df = spark.read.json(input_json_business, schema=business_schema)

    price_range_udf = functions.UserDefinedFunction(lambda attributes: get_price_range(attributes), types.IntegerType())
    # Filter all the businesses which are still open in the business_states
    business_df = business_df.filter((business_df['is_open'] == 1) &
                                     business_df['state'].isin(business_states)) \
        .withColumn('pricerange', price_range_udf(business_df['attributes'])) \
        .withColumnRenamed('business_id', 'b_id') \
        .withColumnRenamed('stars', 'b_stars') \
        .select('b_id', 'city', 'state', 'review_count', 'categories', 'postal_code', 'latitude', 'longitude',
                'pricerange', 'b_stars') \
        .repartition(100)

    business_df.write.format("org.apache.spark.sql.cassandra") \
        .options(table=TABLE_BUSINESS, keyspace=KEY_SPACE) \
        .save(mode='append')

    return business_df


# Read and store review JSON file using the schema
def process_review_json(input_json_review, business_df):
    review_schema = types.StructType([
        types.StructField('review_id', types.StringType(), True),
        types.StructField('business_id', types.StringType(), True),
        types.StructField('user_id', types.StringType(), True),
        types.StructField('text', types.StringType(), True),
        types.StructField('stars', types.IntegerType(), True),
        types.StructField('date', types.DateType(), True)
    ])
    reviews_df = spark.read.json(input_json_review, schema=review_schema)
    review_df = reviews_df.join(business_df, business_df['b_id'] == reviews_df['business_id']) \
        .select(reviews_df.review_id,
                reviews_df.business_id,
                reviews_df.user_id,
                reviews_df.text.alias('review'),
                reviews_df.stars.alias('r_stars'),
                reviews_df.date.alias('r_date')) \
        .repartition(100)

    review_df.write.format("org.apache.spark.sql.cassandra") \
        .options(table=TABLE_REVIEW, keyspace=KEY_SPACE) \
        .save(mode='append')


# Read and store user JSON file using the schema
def process_user_json(input_json_user):
    user_schema = types.StructType([
        types.StructField('user_id', types.StringType(), True),
        types.StructField('average_stars', types.DoubleType(), True),
        types.StructField('review_count', types.LongType(), True),
        types.StructField('yelping_since', types.DateType(), True)
    ])
    users_df = spark.read.json(input_json_user, schema=user_schema).repartition(100)

    users_df.write.format("org.apache.spark.sql.cassandra") \
        .options(table=TABLE_USER, keyspace=KEY_SPACE) \
        .save(mode='append')


def main(businesses_json_file_path_arg, review_json_file_path_arg, user_json_file_path_arg):
    cluster = Cluster(cluster_seeds)
    session = cluster.connect()

    session.execute(QUERY_CREATE_KEY_SPACE)
    session.set_keyspace(KEY_SPACE)

    session.execute(QUERY_CREATE_BUSINESS_TABLE)
    session.execute(QUERY_CREATE_REVIEW_TABLE)
    session.execute(QUERY_CREATE_USER_TABLE)

    business_df = process_business_json(businesses_json_file_path_arg)
    process_review_json(review_json_file_path_arg, business_df)
    process_user_json(user_json_file_path_arg)


# Input will be the path of the business, review and user JSON file
if __name__ == '__main__':
    businesses_json_file_path = sys.argv[1]
    review_json_file_path = sys.argv[2]
    user_json_file_path = sys.argv[3]
    main(businesses_json_file_path, review_json_file_path, user_json_file_path)
