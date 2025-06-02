# This script checks for gaps in the data. Say a company A has values in 1998, 1999, 2000 and again from 2003, 2004, etc.exit
# This inconvenience compromises the data structure we want to see how many such companies share this characteristics.


# The final output "comapnies_with_gaps" looks like this

#                         Cusip | Start year | End Year | Gaps | Missing Years
#                         -----------------------------------------------------
# The start year and end year records the first and last observation of each cusip. 

# The idea is to drop the cusips with gaps, IFF there are minimal number of companies having them.


######################################################################################################################
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min, max, collect_set, array_sort, size, expr, array_distinct, when, sort_array
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql.functions import when, size, lit
from Scripts.functions.create_labelled_data.create_clean_panel import clean_annual_panel
######################################################################################################################

spark = SparkSession.builder.appName("US1").getOrCreate()

input = snakemake.input[0]

df_annual = spark.read.csv(input, header=True, inferSchema=True) 
df_annual = clean_annual_panel(df_annual)

######################################################################################################################


agg_df = df_annual.groupBy("cusip").agg(
    min("fyear").alias("start_year"),
    max("fyear").alias("end_year"),
    array_sort(array_distinct(collect_set("fyear"))).alias("fyear_list")
)

agg_df = agg_df.withColumn(
    "expected_years", expr("sequence(start_year, end_year)")
)

agg_df = agg_df.withColumn(
    "missing_years", expr("array_except(expected_years, fyear_list)")
).withColumn(
    "gaps", size("missing_years") > 0
).withColumn(
    "missing_years", when(size("missing_years") == 0, lit(None)).otherwise(col("missing_years"))
)

final_df = agg_df.select(
    "cusip", "start_year", "end_year", "gaps", "missing_years"
)

######################################################################################################################

# SORTING THE DATAFRAME BY GAPS, TO SEE THE COMPANIES WITH GAPS FIRST

final_df = final_df.orderBy("gaps")

######################################################################################################################

# PERCENTAGE OF COMPANIES HAVING GAPS 

total_count = final_df.count()

no_gap_count = final_df.filter(col("missing_years").isNull()).count()

percent_no_gaps = (no_gap_count / total_count) * 100

print(f"Percentage of observations with no missing years: {percent_no_gaps:.2f}%")

gap_count = final_df.filter(col("missing_years").isNotNull()).count()
print(f"Number of cusips with gaps: {gap_count}")

######################################################################################################################

# SAVING THE RESULTS

final_df = final_df.toPandas()

final_df.to_csv(snakemake.output[0], index = False)

spark.stop()