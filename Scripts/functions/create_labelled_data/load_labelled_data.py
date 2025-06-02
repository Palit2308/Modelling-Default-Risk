#######################################################################################################
from Scripts.functions.create_labelled_data.create_clean_panel import clean_annual_panel
from Scripts.functions.create_labelled_data.remove_gaps import remove_gaps
from Scripts.functions.create_labelled_data.drop_selected_out_of_range_cusips import drop_selected_out_of_range_cusips
from Scripts.functions.create_labelled_data.create_df_matches import create_df_matching
from Scripts.functions.create_labelled_data.create_label_default_status import create_label_default_status
from Scripts.functions.create_labelled_data.dropping_unalabelled_data import drop_unlabelled_data
from pyspark.sql.functions import col
#######################################################################################################

def load_labelled_data(spark, input_fs, input_bk):

    df_annual = spark.read.csv(input_fs, header=True, inferSchema=True) 
    df_cr = spark.read.csv(input_bk, header =True, inferSchema=True)
    cols_to_keep = ["conm", "tic", "cusip", "datadate", "fyear", "spcsrc", "gsector", "at", "ceq", "ch",
                    "che", "chs", "dd1", "dd2", "dd3", "dd4", "dd5", "dt", "bkvlps", "lct", "dvpd", "ebitda",
                    "ebit", "gla", "tie", "sale", "wcapch", "csho", "ajex", "apdedate", "dltt", "dxd2", "dxd3",
                    "dxd4", "dxd5", "invt", "lt", "gp", "tii", "prcc_c"
                   ]
    df_annual = df_annual.select([col(c) for c in df_annual.columns if c in cols_to_keep])
    df_annual = clean_annual_panel(df_annual)
    df_annual = remove_gaps(df_annual)
    df_matches = create_df_matching(spark, df_annual, df_cr)
    df_annual = drop_selected_out_of_range_cusips(df_annual, df_matches)
    df_annual = create_label_default_status(df_annual, df_matches)
    df_annual = drop_unlabelled_data(df_annual)

    return df_annual
