#######################################################################################################
from pyspark.sql.window import Window
from pyspark.sql.functions import to_date, col, year, countDistinct, sum, when, count, lit, round, lag
import pandas as pd
from pyspark.sql.functions import col, lower,row_number
from pyspark.sql.functions import year, count, col, expr, greatest
from pyspark.sql import functions as F
from pyspark.sql.functions import col, to_date, concat_ws
from pyspark.sql.functions import col, when, sequence, explode, lit, min, max
#######################################################################################################

def create_label_default_status(df_annual, df_matches):

    #######################################################################################################
    df_defaults = df_matches.withColumnRenamed("YearFiled", "default_year")
    #######################################################################################################
    joined1 = df_annual.join(df_defaults.select("cusip", "default_year"), on="cusip", how="inner")
    #######################################################################################################
    summary = joined1.groupBy("cusip", "default_year").agg(
        min("fyear").alias("start_year"),
        max("fyear").alias("end_year")
    )
    summary = summary.withColumn(
        "in_range",
        (col("default_year") >= col("start_year")) & (col("default_year") <= col("end_year"))
    )
    #######################################################################################################
    in_range_df = summary.filter(col("in_range") == True)
    in_range_df = in_range_df.withColumn(
        "distance", col("end_year") - col("default_year")
    )

    #######################################################################################################
    out_of_range_df = summary.filter(col("in_range") == False)
    out_of_range_df = out_of_range_df.withColumn(
        "distance", col("default_year") - col("end_year")
    )

    df_matches_trim = df_matches.select("cusip", "YearEmerged", "YearRefiled")
    in_range_enriched = in_range_df.join(df_matches_trim, on="cusip", how="left")

    
    # Labeling strategy for in-range defaults
    def_label_ranges = (
        in_range_enriched
        .withColumn("start1", col("default_year"))
        .withColumn(
            "end1",
            when(col("distance") == 0, col("end_year"))  # label only end year
            .when(col("YearEmerged").isNull(), col("end_year"))  # label full range
            .when(col("YearEmerged") == col("default_year"), col("default_year"))  # label only default year
            .otherwise(col("YearEmerged") - 1)  # label from default_year to YearEmerged - 1
        )
        .withColumn(
            "start2",
            when(
                (col("YearEmerged").isNotNull()) & 
                (col("YearRefiled").isNotNull()) & 
                (col("distance") > 0),
                col("YearRefiled")
            )
        )
        .withColumn(
            "end2",
            when((col("YearEmerged").isNotNull()) & (col("YearRefiled").isNotNull()), col("end_year"))
        )
    )

    # Generate fyear ranges where to label "D"
    ranges1 = def_label_ranges.withColumn("fyear", explode(sequence("start1", "end1"))).select("cusip", "fyear")
    ranges2 = def_label_ranges.withColumn("fyear", explode(sequence("start2", "end2"))).select("cusip", "fyear")
    label_inrange = ranges1.union(ranges2).dropna()

    # For out-of-range: mark only the end year as default
    label_oor = out_of_range_df.select(col("cusip"), col("end_year").alias("fyear"))

    # Combine all labels
    df_labeled = label_inrange.union(label_oor).distinct().withColumn("default_status", lit("D"))

    # Join back with df_annual
    df_final = df_annual.join(df_labeled, on=["cusip", "fyear"], how="left")

    return df_final
