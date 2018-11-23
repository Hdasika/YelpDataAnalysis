import sys
import pyspark
from cassandra import ConsistencyLevel
from cassandra.cluster import Cluster, Session
from cassandra.query import BatchStatement
from pyspark.sql import SparkSession,functions,types
from pyspark.sql.functions import UserDefinedFunction
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

spark = SparkSession.builder.appName('CleanIncomeDataset').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3'  # make sure we have Spark 2.3+

#IMP**
replace = {'arizona': 'az', 'pennsylvania': 'pa', 'nevada': 'nv', 'north carolina': 'nc',
           'ohio': 'oh', 'illinois': 'il'}

key_space = 'hdasika'
income_table = "CREATE TABLE IF NOT EXISTS review ( review_id TEXT PRIMARY KEY, " \
                            "zip_code INT, " \
                            "State TEXT, " \
                            "County TEXT, " \
                            "income DOUBLE );"

def main(income_csv,state_csv):
    cluster = Cluster()
    session = cluster.connect(key_space)
    session.execute(income_table)
    incomedf=process_income_csv(income_csv)
    statedf= process_state_csv(state_csv)
    incomedf.write.csv('income.csv') # writing as CSV in between
    income_states = clean_income_zip(statedf, incomedf,session)
    income_states.write.csv('income.csv')


def clean_income_zip(statesdf, incomedf,session: Session):
    Merged=statesdf.join(incomedf,statesdf['statCombine']==incomedf['incomeCombine','Left'])
    z=Merged.na.drop() # ----->is it required??
    incomezip=Merged.select('zip_code','State','County','combine','income')
    insert_user_statement = session.prepare(
        "INSERT INTO review ('zip_code','State','County','combine','income') "
        "VALUES (?, ?, ?, ?, ?)")
    batch = BatchStatement(consistency_level=ConsistencyLevel.ONE)

    bath_size = 0
    for row in incomezip.rdd.collect():
        batch.add(insert_user_statement, (row['zip_code'],
                                          row['State'],
                                          row['County'],
                                          row['combine'],
                                          row['income'],
                                          )
                  )
        bath_size += 1
        if bath_size >= 100:
            session.execute(batch)
            bath_size = 0
            batch.clear()
    session.execute(batch)
    return incomezip



def process_state_csv(state_csv):
    state = spark.read.format('CSV').option("header", "true").load(state_csv)
    State=state.select('zip_code','state','county')
    A=State.withColumn('State', functions.lower(State['state']))
    B=A.withColumn('County', functions.lower(State['county']))
    StatCombine=B.withColumn('statCombine',functions.concat(B.State,A.County))
    StatCombine.select('statCombine','zip_code','County','State')
    return StatCombine

def process_income_csv(income_csv):
    #state_df = spark.read.csv(state_csv, schema=state_schema)
    #income = spark.read.format('CSV').option("header", "true").load(income_csv) -----> which is better??
    #_df = income.select(income['GEO.display-label'],income['HC01_EST_VC02'],income['HC01_MOE_VC02'],income['HC02_EST_VC02']).repartition(100)
    income = spark.read.csv(income_csv)
    x = income.select(income['_c2'].alias('label'),income['_c5'].alias('income'))
    y=x.filter(x.label!='GEO.display-label') #alternative for removing
    split_col = pyspark.sql.functions.split(y['label'], ',')
    df=y.withColumn('County1', functions.lower(split_col.getItem(0)))
    df1 = df.withColumn('State', functions.lower(split_col.getItem(1)))
    split_col1 = pyspark.sql.functions.split(y['County1'], ' ')
    df2 = df1.withColumn('County', functions.lower(split_col.getItem(0))).drop('label').drop('County1')
    #split_col2 = pyspark.sql.functions.split(df2['State'], '')
    #z = df2.withColumn('StateCut', functions.lower(df2['State'][1:3]))
    udf = functions.UserDefinedFunction(lambda x: replace.get(x), types.StringType()) #---- Changed after speaking
    out = df2.withColumn("StateCut", udf(df2["State"])) #changed after speaking
    fil_df=out.filter(out['State'].asin([*replace])) # Will filter the states need to check
    combine=fil_df.withColumn('incomeCombine',functions.concat(fil_df.StateCut,fil_df.County)).select('incomeCombine','income')
    return combine


if __name__ == '__main__':
    income_csv = sys.argv[1]
    state_csv = sys.argv[2]
    user_json_file_path = sys.argv[3]
    main(income_csv,state_csv)