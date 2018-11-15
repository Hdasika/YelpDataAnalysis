import sys

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

from pyspark.sql import SparkSession, types

cluster_seeds = ['199.60.17.188', '199.60.17.216']
spark = SparkSession.builder.appName('CleanYelpDataset') \
    .config('spark.cassandra.connection.host', ','.join(cluster_seeds)).getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3'  # make sure we have Spark 2.3+

zipcode_states_schema = types.StructType([
    types.StructField('zip_code', types.IntegerType(), True),
    types.StructField('latitude', types.DoubleType(), True),
    types.StructField('longitude', types.DoubleType(), True),
    types.StructField('city', types.StringType(), True),
    types.StructField('state', types.StringType(), True),
    types.StructField('county', types.StringType(), True)
])


def clean_zipcode_states(zip_code_states_file_path_arg):
    zip_code_states_df = spark.read.csv(zip_code_states_file_path_arg, schema=zipcode_states_schema)
    zip_code_states_df.show()
    # select information according to each state
    # az = df[df['state'] == 'AZ']
    # pa = df[df['state'] == 'PA']
    # nv = df[df['state'] == 'NV']
    # nc = df[df['state'] == 'NC']
    # il = df[df['state'] == 'IL']
    # oh = df[df['state'] == 'OH']
    # # concatenate together
    # states_df = pd.concat([az, pa, nv, nc, oh, il])
    # # select useful columns only
    # states_df = states_df.iloc[:, [0, 4, 5]]
    # # lower case all letters
    # states_df['state'] = states_df['state'].str.lower()
    # states_df['county'] = states_df['county'].str.lower()
    # # create a column that combine state and county name
    # states_df['combine'] = states_df['county'] + states_df['state']
    return zip_code_states_df


# Input will be the path of the zip_code_states, review and user JSON file
if __name__ == '__main__':
    zip_code_states_file_path = sys.argv[1]
    # income_file_path = sys.argv[2]
    clean_zipcode_states(zip_code_states_file_path)
