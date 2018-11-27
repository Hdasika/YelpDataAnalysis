import sys

import pyspark
from cassandra.cluster import Cluster
from pyspark.sql import SparkSession, types, functions

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

cluster_seeds = ['199.60.17.188', '199.60.17.216', '127.0.0.1']
spark = SparkSession.builder \
    .config('spark.cassandra.connection.host', ','.join(cluster_seeds)) \
    .appName('CleanYelpDataset') \
    .getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3'  # make sure we have Spark 2.3+

business_states = ['PA', 'NV', 'NC', 'IL', 'OH', 'AZ']
required_states_abbr = ['pa', 'nv', 'nc', 'il', 'oh', 'az']
states_abbr_mapping = {'arizona': 'az',
                       'pennsylvania': 'pa',
                       'nevada': 'nv',
                       'north carolina': 'nc',
                       'ohio': 'oh',
                       'illinois': 'il'}

KEY_SPACE = 'bigp18'
TABLE_BUSINESS = 'business'
TABLE_USER = 'user'
TABLE_REVIEW = 'review'
TABLE_INCOME = 'income'

QUERY_CREATE_KEY_SPACE = "CREATE KEYSPACE IF NOT EXISTS " + KEY_SPACE + \
                         " WITH replication = { 'class' : 'SimpleStrategy', 'replication_factor' : 3 };"

QUERY_CREATE_BUSINESS_TABLE = "CREATE TABLE IF NOT EXISTS " + TABLE_BUSINESS + " ( b_id TEXT PRIMARY KEY, " \
                                                                               "name TEXT, " \
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
                                                                           "zip_code INT, " \
                                                                           "user_id TEXT, " \
                                                                           "review TEXT, " \
                                                                           "r_stars INT, " \
                                                                           "r_date DATE );"
QUERY_CREATE_INCOME_TABLE = "CREATE TABLE IF NOT EXISTS " + TABLE_INCOME + " ( zip_code TEXT PRIMARY KEY, " \
                                                                           "state TEXT, " \
                                                                           "county TEXT, " \
                                                                           "avg_income INT );"


def write_to_cassandra(data_frame, table_name):
    data_frame.show()
    data_frame.write.format("org.apache.spark.sql.cassandra") \
        .options(table=table_name, keyspace=KEY_SPACE) \
        .save(mode='append')


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
        .select('b_id', 'name', 'city', 'state', 'review_count', 'categories',
                'postal_code', 'latitude', 'longitude', 'pricerange', 'b_stars') \
        .repartition(100)

    write_to_cassandra(business_df, TABLE_BUSINESS)
    return business_df


# Read and store user JSON file using the schema
def process_user_json(input_json_user):
    user_schema = types.StructType([
        types.StructField('user_id', types.StringType(), True),
        types.StructField('average_stars', types.DoubleType(), True),
        types.StructField('review_count', types.LongType(), True),
        types.StructField('yelping_since', types.DateType(), True)
    ])
    users_df = spark.read.json(input_json_user, schema=user_schema).repartition(100)

    write_to_cassandra(users_df, TABLE_USER)


# Read and store review JSON file using the schema
def process_review_json(input_json_review, business_df, merged_income_df):
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
        .repartition(100)

    review_df = review_df.join(merged_income_df, merged_income_df['zip_code'] == business_df['postal_code']) \
        .select(reviews_df.review_id,
                reviews_df.business_id,
                merged_income_df.zip_code,
                reviews_df.user_id,
                reviews_df.text.alias('review'),
                reviews_df.stars.alias('r_stars'),
                reviews_df.date.alias('r_date'))

    write_to_cassandra(review_df, TABLE_REVIEW)


# Read and temporarily create a income data frame from CSV file of selected US states
def process_income_csv(income_csv_arg):
    income_df = spark.read.csv(income_csv_arg)
    income_df = income_df.filter((income_df['_c2'] != 'GEO.display-label') & (income_df['_c2'] != 'Geography')) \
        .select(income_df['_c2'].alias('county_state'), income_df['_c7'].alias('income'))

    county_state_column = pyspark.sql.functions.split(income_df['county_state'], ',')
    county_name = pyspark.sql.functions.split(county_state_column.getItem(0), ' ').getItem(0)
    state_name = functions.trim(county_state_column.getItem(1))

    income_df = income_df.withColumn('county', pyspark.sql.functions.lower(county_name)) \
        .withColumn('state', functions.lower(state_name)) \
        .withColumn('income', income_df['income'].cast('int')) \
        .drop('county_state')

    state_abbr_udf = functions.UserDefinedFunction(lambda state: states_abbr_mapping.get(state), types.StringType())
    income_df = income_df.withColumn("state_abbr", state_abbr_udf(income_df["state"])).drop('state')
    income_df = income_df.withColumn('combine', functions.concat(income_df['county'], income_df['state_abbr'])) \
        .filter((income_df['state_abbr'].isNotNull()) & (income_df['state_abbr'].isin(required_states_abbr)))

    return income_df


# Read and temporarily create a zip code state data frame from CSV file of selected US states
def process_zip_code_state_csv(zip_code_state_csv_arg):
    zip_code_states_df = spark.read.csv(zip_code_state_csv_arg, header=True)

    zip_code_states_df = zip_code_states_df \
        .withColumn('state', functions.lower(zip_code_states_df['state'])) \
        .withColumn('county', functions.lower(zip_code_states_df['county']))
    zip_code_states_df = zip_code_states_df.withColumn('combine', functions.concat(zip_code_states_df['county'],
                                                                                   zip_code_states_df['state']))
    zip_code_states_df = zip_code_states_df.select('zip_code', 'county', 'state', 'combine') \
        .filter(zip_code_states_df['state'].isin(required_states_abbr))
    return zip_code_states_df


# Merge income zip_code data frames and write into income table
def merge_income_and_zip_code_data(income_df_arg, zip_code_states_df_arg):
    merged_income_and_zip_code_df = zip_code_states_df_arg.join(income_df_arg,
                                                                zip_code_states_df_arg['combine'] ==
                                                                income_df_arg['combine'], how='inner') \
        .select(zip_code_states_df_arg.zip_code,
                zip_code_states_df_arg.state,
                zip_code_states_df_arg.county,
                income_df_arg.income.alias('avg_income'))

    write_to_cassandra(merged_income_and_zip_code_df, TABLE_INCOME)
    return merged_income_and_zip_code_df


def main(businesses_json_file_path_arg,
         review_json_file_path_arg,
         user_json_file_path_arg,
         income_csv_arg,
         zip_code_state_csv_arg):
    cluster = Cluster(cluster_seeds)
    session = cluster.connect()

    session.execute(QUERY_CREATE_KEY_SPACE)
    session.set_keyspace(KEY_SPACE)

    session.execute(QUERY_CREATE_BUSINESS_TABLE)
    session.execute(QUERY_CREATE_REVIEW_TABLE)
    session.execute(QUERY_CREATE_USER_TABLE)
    session.execute(QUERY_CREATE_INCOME_TABLE)

    income_df = process_income_csv(income_csv_arg)
    zip_code_state_df = process_zip_code_state_csv(zip_code_state_csv_arg)
    merged_income_df = merge_income_and_zip_code_data(income_df, zip_code_state_df)

    business_df = process_business_json(businesses_json_file_path_arg)
    process_review_json(review_json_file_path_arg, business_df, merged_income_df)
    process_user_json(user_json_file_path_arg)


# Input will be the path of the business, review and user JSON file
if __name__ == '__main__':
    income_csv_path = sys.argv[1]
    zip_code_state_csv_path = sys.argv[2]
    businesses_json_file_path = sys.argv[3]
    review_json_file_path = sys.argv[4]
    user_json_file_path = sys.argv[5]

    main(businesses_json_file_path,
         review_json_file_path,
         user_json_file_path,
         income_csv_path,
         zip_code_state_csv_path)
