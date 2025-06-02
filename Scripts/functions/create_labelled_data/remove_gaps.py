######################################################################################################################
from pyspark.sql.functions import col, min, max, collect_set, array_sort, size, expr, array_distinct, when, sort_array
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql.functions import when, size, lit

######################################################################################################################

def remove_gaps(df_annual, fyear = "fyear"):
    
    agg_df = df_annual.groupBy("cusip").agg(
    min(fyear).alias("start_year"),
    max(fyear).alias("end_year"),
    array_sort(array_distinct(collect_set(fyear))).alias("fyear_list")
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

    gappy_companies = final_df.filter(col("gaps") == True).select("cusip").distinct()

    # gappy_companies.count()

    df_cleaned = df_annual.join(gappy_companies, on="cusip", how="left_anti")

    return df_cleaned
