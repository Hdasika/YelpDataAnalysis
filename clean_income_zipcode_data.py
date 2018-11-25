import sys

import pyspark
from cassandra import ConsistencyLevel
from cassandra.cluster import Cluster, Session
from cassandra.query import BatchStatement
from pyspark.sql import SparkSession, functions, types, DataFrame

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

spark = SparkSession.builder.appName('CleanIncomeDataset').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3'  # make sure we have Spark 2.3+

required_states_abbr = ['az', 'pa', 'nv', 'nc', 'oh', 'il']
states_abbr_mapping = {'arizona': 'az',
                       'pennsylvania': 'pa',
                       'nevada': 'nv',
                       'north carolina': 'nc',
                       'ohio': 'oh',
                       'illinois': 'il'}

key_space = 'bigp18'
query_create_income_table = "CREATE TABLE IF NOT EXISTS income ( zip_code TEXT PRIMARY KEY, " \
                            "state TEXT, " \
                            "county TEXT, " \
                            "combine TEXT, " \
                            "avg_income INT );"


def process_income_csv(income_csv_arg):
    income_df = spark.read.csv(income_csv_arg)
    income_df = income_df.filter(income_df['_c2'] != 'GEO.display-label') \
        .select(income_df['_c2'].alias('county_state'), income_df['_c5'].alias('income'))

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


def merge_income_and_zip_code_data(income_df_arg: DataFrame, zip_code_states_df_arg: DataFrame, session: Session):
    merged_income_and_zip_code_df = zip_code_states_df_arg.join(income_df_arg,
                                                                zip_code_states_df_arg['combine'] ==
                                                                income_df_arg['combine'], how='left') \
        .select(zip_code_states_df_arg.zip_code,
                zip_code_states_df_arg.state,
                zip_code_states_df_arg.county,
                zip_code_states_df_arg.combine,
                income_df_arg.income)
    merged_income_and_zip_code_df.show()

    insert_user_statement = session.prepare(
        "INSERT INTO income (zip_code, state, county, combine, avg_income) "
        "VALUES (?, ?, ?, ?, ?)")
    batch = BatchStatement(consistency_level=ConsistencyLevel.ONE)

    bath_size = 0
    for row in merged_income_and_zip_code_df.rdd.collect():
        batch.add(insert_user_statement, (row['zip_code'],
                                          row['state'],
                                          row['county'],
                                          row['combine'],
                                          row['income'])
                  )
        bath_size += 1
        if bath_size >= 100:
            session.execute(batch)
            bath_size = 0
            batch.clear()
    session.execute(batch)

    return merged_income_and_zip_code_df


def main(income_csv_arg, zip_code_state_csv_arg, merged_csv_path_arg):
    cluster = Cluster()
    session = cluster.connect(key_space)
    session.execute(query_create_income_table)

    income_df = process_income_csv(income_csv_arg)
    zip_code_state_df = process_zip_code_state_csv(zip_code_state_csv_arg)

    merge_income_and_zip_code_data(income_df, zip_code_state_df, session)


if __name__ == '__main__':
    income_csv = sys.argv[1]
    zip_code_state_csv = sys.argv[2]
    main(income_csv, zip_code_state_csv)
