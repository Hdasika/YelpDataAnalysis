import sys

from cassandra import ConsistencyLevel
from cassandra.cluster import Cluster, Session
from cassandra.query import BatchStatement
from pyspark.sql import SparkSession, types, DataFrame, functions

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

spark = SparkSession.builder \
    .config("spark.executor.memory", "70g") \
    .config("spark.driver.memory", "50g") \
    .config("spark.memory.offHeap.enabled", True) \
    .config("spark.memory.offHeap.size", "16g") \
    .appName('CleanYelpDataset') \
    .getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3'  # make sure we have Spark 2.3+

business_states = ['ON', 'QC', 'NS', 'NB', 'MB', 'BC', 'PE', 'SK', 'AB', 'NL']

key_space = 'bigp18'
query_create_business_table = "CREATE TABLE IF NOT EXISTS business ( b_id TEXT PRIMARY KEY, " \
                              "city TEXT, " \
                              "state TEXT, " \
                              "review_count INT, " \
                              "categories TEXT, " \
                              "postal_code TEXT, " \
                              "latitude DOUBLE, " \
                              "longitude DOUBLE, " \
                              "pricerange INT, " \
                              "b_stars FLOAT );"
query_create_user_table = "CREATE TABLE IF NOT EXISTS users ( user_id TEXT PRIMARY KEY, " \
                          "review_count INT, " \
                          "yelping_since DATE, " \
                          "average_stars FLOAT );"
query_create_review_table = "CREATE TABLE IF NOT EXISTS review ( review_id TEXT PRIMARY KEY, " \
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
def process_business_json(input_json_business, session: Session):
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
        .repartition(100)

    insert_business_statement = session.prepare(
        "INSERT INTO business (b_id, city, state, review_count, categories, "
        "postal_code, latitude, longitude, pricerange, b_stars) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
    batch = BatchStatement(consistency_level=ConsistencyLevel.ONE)

    bath_size = 0
    for row in business_df.rdd.collect():
        batch.add(insert_business_statement, (row['business_id'],
                                              row['city'],
                                              row['state'],
                                              row['review_count'],
                                              row['categories'],
                                              row['postal_code'],
                                              row['latitude'],
                                              row['longitude'],
                                              row['pricerange'],
                                              row['stars'])
                  )
        bath_size += 1

        if bath_size >= 100:
            session.execute(batch)
            bath_size = 0
            batch.clear()
    session.execute(batch)
    return business_df


# Read and store review JSON file using the schema
def process_review_json(input_json_review, business_df: DataFrame, session: Session):
    review_schema = types.StructType([
        types.StructField('review_id', types.StringType(), True),
        types.StructField('business_id', types.StringType(), True),
        types.StructField('user_id', types.StringType(), True),
        types.StructField('text', types.StringType(), True),
        types.StructField('stars', types.IntegerType(), True),
        types.StructField('date', types.DateType(), True)
    ])
    reviews_df = spark.read.json(input_json_review, schema=review_schema)
    review_df = reviews_df.join(business_df, business_df['business_id'] == reviews_df['business_id']) \
        .select(reviews_df.review_id,
                reviews_df.business_id,
                reviews_df.user_id,
                reviews_df.text,
                reviews_df.stars,
                reviews_df.date) \
        .repartition(100)

    insert_user_statement = session.prepare(
        "INSERT INTO review (review_id, business_id, user_id, review, r_stars, r_date) "
        "VALUES (?, ?, ?, ?, ?, ?)")
    batch = BatchStatement(consistency_level=ConsistencyLevel.ONE)

    bath_size = 0
    for row in review_df.rdd.collect():
        batch.add(insert_user_statement, (row['review_id'],
                                          row['business_id'],
                                          row['user_id'],
                                          row['text'],
                                          row['stars'],
                                          row['date'])
                  )
        bath_size += 1

        if bath_size >= 100:
            session.execute(batch)
            bath_size = 0
            batch.clear()
    session.execute(batch)


# Read and store user JSON file using the schema
def process_user_json(input_json_user, session: Session):
    user_schema = types.StructType([
        types.StructField('user_id', types.StringType(), True),
        types.StructField('average_stars', types.DoubleType(), True),
        types.StructField('review_count', types.LongType(), True),
        types.StructField('yelping_since', types.DateType(), True)
    ])
    users_df = spark.read.json(input_json_user, schema=user_schema).repartition(100)

    insert_user_statement = session.prepare("INSERT INTO users (user_id, review_count, yelping_since, average_stars) "
                                            "VALUES (?, ?, ?, ?)")
    batch = BatchStatement(consistency_level=ConsistencyLevel.ONE)

    bath_size = 0
    for row in users_df.rdd.collect():
        batch.add(insert_user_statement, (row['user_id'],
                                          row['review_count'],
                                          row['yelping_since'],
                                          row['average_stars'])
                  )
        bath_size += 1

        if bath_size >= 100:
            session.execute(batch)
            bath_size = 0
            batch.clear()
    session.execute(batch)


def main(businesses_json_file_path_arg, review_json_file_path_arg, user_json_file_path_arg):
    cluster = Cluster()
    session = cluster.connect(key_space)

    session.execute(query_create_business_table)
    session.execute(query_create_review_table)
    session.execute(query_create_user_table)

    business_df = process_business_json(businesses_json_file_path_arg, session)
    process_review_json(review_json_file_path_arg, business_df, session)
    process_user_json(user_json_file_path_arg, session)


# Input will be the path of the business, review and user JSON file
if __name__ == '__main__':
    businesses_json_file_path = sys.argv[1]
    review_json_file_path = sys.argv[2]
    user_json_file_path = sys.argv[3]
    main(businesses_json_file_path, review_json_file_path, user_json_file_path)
