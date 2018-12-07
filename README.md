Readme
======
Code Functionality
------------------
Part 1 -> Table Creations (ETL)
-------------------------------
The code does generate four tables and load into casssandra cluster 
The main filter applied is , the code ran is specific to 7 particular states.['WI','PA', 'NV', 'NC', 'IL', 'OH', 'AZ']
The below is the gist on the operations performed to get the tables up and live in cassandra cluster.

--->'user' table
- Picks the user.json from the yelp dataset and as per requirement process the data 
and outputs are stored in user table in cassandra with columns ('user_id','average_stars','review_count','yelping_since')

---> 'business' table
- Picks the business.json file from the yelp dataset and filtering the required fields as in 
('b_id', 'name', 'city', 'state', 'review_count', 'categories',
                'postal_code', 'latitude', 'longitude', 'pricerange', 'b_stars') as per needs of the plan
   and stores in Cassandra db.
->Outputs a Business_Df

--->'review' table
- Picks the review.json from yelp dataset per requirement filters out the needed columns and ouputs a Df.
This Df is joined with the business_df(generate earlier) and outputs a df after processing the data.
A table is generated at the end in the cassandra table name 'review' with the columns(review_id,business_id,
zip_code,user_id,review,r_stars,r_date)

--->'income' table
Two parts are involved over here (involves joining income.csv and zip_code_state.csv data)
- The household income data is picked from income.csv where in only two(county_state,income) columns are picked. The County_state has been combined 
 (ex: Baldwin County,Alabama = baldwinal) forming a column 'combine'. After this an 'income_df' is obtained.
- The zip_code_state.csv has got state and county which is proccessed in a way as above(ex: Baldwin County,Alabama = baldwinal)
 and named it as column 'combine', this particular dataframe and the above obtained income_df is joined on the column 'combine'
 and then a table with name 'income' and columns (zip_code,state,county,avg_income) is created in Cassandra db
 

Part - 2 (Fake Review & Word Cloud)
--------
Part - 3 (Web Frontend -- Visualization)
--------
Tableau WorkBook --- Significance



Runners
=======


|Extract Transform Logic (ETL)|
-----------------------------> 
To Run the code User needs to provide
Prerequisite :-
- PySpark 2.3+
- Python 3
- Cassandra connection
- Datasets:- review.json,user.json,business.json , income.csv,zip_codes_states.csv
- path(datasource) = /user/sfuid/yelp/
==>Commands to generate the tables in cassandra 
"""spark-submit --packages datastax:spark-cassandra-connector:2.3.1-s_2.11 
 clean_yelp_dataset.py path/income.csv path/zip_codes_states.csv path/business.json path/review.json path/user.json"""
 
|Fake Review |
------------->
 Fake Review
 ==>Commands