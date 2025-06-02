######################################################################################################################
from pyspark.sql.window import Window
from pyspark.sql.functions import col, min, max, collect_set, array_sort, size, expr, array_distinct, when, sort_array
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql.functions import when, size, lit
from pyspark.sql.functions import col, min, max, to_date,year, lower, row_number
from pyspark.sql.functions import col, expr
######################################################################################################################

def create_df_matching(spark, df_annual, df_cr):
    # CREATE THE DF_MATCHES ------  CUSIP | YEARFILED | CHAPTER | YEAREMERGED | YEARREFILED
    #######################################################################################################
    df_cr = df_cr.filter((col("Chapter") == "7") | (col("Chapter") == "11"))
    #######################################################################################################
    df_cr= df_cr.filter(col("DateFiled").isNotNull())
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
    df_cr = df_cr.withColumn("DateFiled", to_date("DateFiled", "M/d/yyyy"))
    df_cr = df_cr.withColumn("DateEmerging", to_date("DateEmerging", "M/d/yyyy"))
    df_cr = df_cr.withColumn("DateRefile", to_date("DateRefile", "M/d/yyyy"))
    #######################################################################################################
    df_cr = df_cr.withColumn("YearFiled", year(col("DateFiled")))
    df_cr = df_cr.withColumn("YearEmerged", year(col("DateEmerging")))
    df_cr = df_cr.withColumn("YearRefiled", year(col("DateRefile")))
    #######################################################################################################
    df_cr = df_cr.select("NameCorp", "Chapter", "DateFiled", "cusip", "YearFiled", "YearEmerged", "YearRefiled")
    #######################################################################################################
    df_annual_names = df_annual["conm", "tic", "cusip"]
    df_unique_conms = df_annual_names.select("cusip").distinct()
    #######################################################################################################
    df_matches = df_cr.alias("cr").join(
        df_unique_conms.alias("an"),
        (lower(col("cr.cusip")).contains(lower(col("an.cusip")))) |
        (lower(col("an.cusip")).contains(lower(col("cr.cusip"))))
    )
    #######################################################################################################
    window_spec = Window.partitionBy("an.cusip").orderBy(col("YearFiled").desc())
    df_matches = df_matches.withColumn("row_num", row_number().over(window_spec))
    df_matches = df_matches.filter(col("row_num") == 1).drop("row_num")
    df_matches = df_matches.select(
        col("an.cusip").alias("cusip"),
        col("cr.NameCorp"),
        col("cr.YearFiled"),
        col("cr.Chapter"),
        col("cr.YearEmerged"),
        col("cr.YearRefiled")
    )
    
    return df_matches
