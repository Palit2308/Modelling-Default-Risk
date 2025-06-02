from pyspark.sql import SparkSession
from Scripts.functions.create_labelled_data.load_labelled_data import load_labelled_data
from pyspark.sql.functions import col, lit, sum as spark_sum, round as spark_round
spark = SparkSession.builder.appName("US1").getOrCreate()

input1 = snakemake.input[0]
input2 = snakemake.input[1]
output = snakemake.output[0]


df_annual = load_labelled_data(spark, input1, input2)

df_desc = df_annual.select("cusip", "fyear", "at", "ch", "ceq", "csho", "lct", "dd1", "dltt", "dt", "gp", "ebit", "sale", "default_status", "prcc_c")

df_desc.toPandas().to_csv("Data/prepared_datasets/financials_annual_merton.csv", index = False)