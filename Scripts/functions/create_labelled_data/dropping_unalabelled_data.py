#######################################################################################################
from pyspark.sql import Window
from pyspark.sql.functions import col, lag, countDistinct, count
from pyspark.sql.functions import col, lit, sum as spark_sum, round as spark_round
from pyspark.sql import functions as F
#######################################################################################################

def drop_unlabelled_data(df_annual):

    cusips_all_null = df_annual.groupBy("cusip").agg(
        F.max(F.col("default_status").isNotNull().cast("int")).alias("has_non_null")
    ).filter("has_non_null = 0")
    #######################################################################################################

    valid_spcsrc_values = ["C", "B+", "B-", "B", "A-", "D", "A+", "LIQ", "A"]

    filtered_df = df_annual.join(cusips_all_null.select("cusip").distinct(), on="cusip", how="inner") \
                            .filter(F.col("spcsrc").isin(valid_spcsrc_values))
    #######################################################################################################

    cusips_to_keep = filtered_df.select("cusip").distinct()
    cusips_not_in_excluded = df_annual.select("cusip").distinct().join(cusips_all_null, on="cusip", how="left_anti")
    #######################################################################################################

    final_df = df_annual.join(cusips_to_keep.union(cusips_not_in_excluded), on="cusip", how="inner")

    return final_df
