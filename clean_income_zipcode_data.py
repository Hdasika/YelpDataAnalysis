import sys

import pyspark
from cassandra.cluster import Cluster, Session
from pyspark.sql import SparkSession, functions, types, DataFrame

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

cluster_seeds = ['199.60.17.188', '199.60.17.216']
spark = SparkSession.builder \
    .config('spark.cassandra.connection.host', ','.join(cluster_seeds)) \
    .appName('CleanIncomeDataset') \
    .getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3'  # make sure we have Spark 2.3+

required_states_abbr = ['pa', 'nv', 'nc', 'il', 'oh', 'az']
states_abbr_mapping = {'arizona': 'az',
                       'pennsylvania': 'pa',
                       'nevada': 'nv',
                       'north carolina': 'nc',
                       'ohio': 'oh',
                       'illinois': 'il'}

KEY_SPACE = 'bigp18'
TABLE_INCOME = 'income'

QUERY_CREATE_KEY_SPACE = "CREATE KEYSPACE IF NOT EXISTS " + KEY_SPACE + \
                         " WITH replication = { 'class' : 'SimpleStrategy', 'replication_factor' : 3 };"

QUERY_CREATE_INCOME_TABLE = "CREATE TABLE IF NOT EXISTS " + TABLE_INCOME + " ( zip_code TEXT PRIMARY KEY, " \
                                                                           "state TEXT, " \
                                                                           "county TEXT, " \
                                                                           "combine TEXT, " \
                                                                           "avg_income INT );"


def process_income_csv(income_csv_arg):
    income_df = spark.read.csv(income_csv_arg)
    income_df = income_df.filter((income_df['_c2'] != 'GEO.display-label') | (income_df['_c2'] != 'Geography')) \
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
        .filter(income_df['state_abbr'].isin(required_states_abbr))
    return income_df


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


def merge_income_and_zip_code_data(income_df_arg, zip_code_states_df_arg):
    merged_income_and_zip_code_df = zip_code_states_df_arg.join(income_df_arg,
                                                                zip_code_states_df_arg['combine'] ==
                                                                income_df_arg['combine'], how='inner') \
        .select(zip_code_states_df_arg.zip_code,
                zip_code_states_df_arg.state,
                zip_code_states_df_arg.county,
                zip_code_states_df_arg.combine,
                income_df_arg.income.alias('avg_income'))

    merged_income_and_zip_code_df.write.format("org.apache.spark.sql.cassandra") \
        .options(table=TABLE_INCOME, keyspace=KEY_SPACE) \
        .save(mode='append')


def main(income_csv_arg, zip_code_state_csv_arg):
    cluster = Cluster(cluster_seeds)
    session = cluster.connect()

    session.execute(QUERY_CREATE_KEY_SPACE)
    session.set_keyspace(KEY_SPACE)
    session.execute(QUERY_CREATE_INCOME_TABLE)

    income_df = process_income_csv(income_csv_arg)
    zip_code_state_df = process_zip_code_state_csv(zip_code_state_csv_arg)

    merge_income_and_zip_code_data(income_df, zip_code_state_df)


if __name__ == '__main__':
    income_csv = sys.argv[1]
    zip_code_state_csv = sys.argv[2]
    main(income_csv, zip_code_state_csv)
