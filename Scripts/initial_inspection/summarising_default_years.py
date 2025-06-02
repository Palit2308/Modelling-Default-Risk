######################################################################################################################
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min, max, collect_set, array_sort, size, expr, array_distinct, when, sort_array
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql.functions import when, size, lit
from Scripts.functions.create_labelled_data.create_clean_panel import clean_annual_panel
from Scripts.functions.create_labelled_data.remove_gaps import remove_gaps
from pyspark.sql.functions import col, min, max
from pyspark.sql.functions import col, expr
######################################################################################################################

spark = SparkSession.builder.appName("US1").getOrCreate()

input1 = snakemake.input[0]
input2 = snakemake.input[1]

df_annual = spark.read.csv(input1, header=True, inferSchema=True) 
df_annual = clean_annual_panel(df_annual)
df_annual = remove_gaps(df_annual)
df_default_data = spark.read.csv(input2, header =True, inferSchema=True)

######################################################################################################################

df_defaults = df_default_data.withColumnRenamed("YearFiled", "default_year")

joined = df_annual.join(df_defaults.select("cusip", "default_year"), on="cusip", how="inner")

summary = joined.groupBy("cusip", "default_year").agg(
    min("fyear").alias("start_year"),
    max("fyear").alias("end_year")
)

summary = summary.withColumn(
    "in_range",
    (col("default_year") >= col("start_year")) & (col("default_year") <= col("end_year"))
)

######################################################################################################################

summary_of_summary = summary.groupBy("in_range").count()

######################################################################################################################

out_of_range_df = summary.filter(col("in_range") == False)

out_of_range_df = out_of_range_df.withColumn(
    "distance", col("default_year") - col("end_year")
)

distance_summary_out_of_range = out_of_range_df.groupBy("distance").count().orderBy("distance")

######################################################################################################################

in_range_df = summary.filter(col("in_range") == True)

in_range_df = in_range_df.withColumn(
    "distance", col("end_year") - col("default_year")
)

distance_summary_in_range = in_range_df.groupBy("distance").count().orderBy("distance")

######################################################################################################################

summary = summary.toPandas()
summary_of_summary = summary_of_summary.toPandas()
distance_summary_out_of_range = distance_summary_out_of_range.toPandas()
distance_summary_in_range = distance_summary_in_range.toPandas()

summary.to_csv(snakemake.output[0], index = False)
summary_of_summary.to_csv(snakemake.output[1], index = False)
distance_summary_out_of_range.to_csv(snakemake.output[2], index = False)
distance_summary_in_range.to_csv(snakemake.output[3], index = False)







# # Step 1: Identify the good cusips (distance == 1, 2, or 3)
# good_out_of_range_cusips = out_of_range_df \
#     .filter(col("distance").isin([1, 2, 3])) \
#     .select("cusip") \
#     .distinct()

# # Step 2: Identify all out-of-range cusips (the full set)
# all_out_of_range_cusips = out_of_range_df.select("cusip").distinct()

# # Step 3: Find the BAD ones (those NOT in the good set)
# bad_out_of_range_cusips = all_out_of_range_cusips.join(
#     good_out_of_range_cusips, on="cusip", how="left_anti"
# )

# # Step 4: Remove BAD cusips from df_annual
# df_annual_filtered = df_annual.join(bad_out_of_range_cusips, on="cusip", how="left_anti")

# df_annual_filtered.select("cusip").distinct().count()
