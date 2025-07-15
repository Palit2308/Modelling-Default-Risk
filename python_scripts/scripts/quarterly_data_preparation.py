import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from python_scripts.functions.create_reqd_columns import create_reqd_colums
from python_scripts.functions.merton_model import merton_solver, distance_to_default, two_system_merton
from python_scripts.functions.iterative_svd_impute import iterative_svd_impute
from python_scripts.functions.compute_auc_pauc import compute_auc_pauc

# Taking the quarterly data from 2006 Q1 to 2025 Q2

df = pd.read_csv(snakemake.input[0], compression = "gzip")

df1 = df[["cusip", "fyearq", "fqtr", "prccq", "mkvaltq", "xrdq", "xoprq",
          "xintq", "xsgaq", "txwq", "ajexq", "atq", "dlcq", "dlttq", "gdwlq",
          "glaq", "glpq", "invrmq", "invtq", "lctq", "optrfrq", "optvolq", "revtq",
          "saleq", "cshoq", "spcsrc", "gind", "cogsq", "dd1q"]]

# df1.describe().T

df1 = df1.set_index(["cusip", "fyearq", "fqtr"])

# df_norating = df1["spcsrc"].value_counts(dropna=False)

# Dropping unrated observations

df1 = df1.dropna(subset = ["spcsrc"])

# df1["spcsrc"].value_counts()

##################################################################################################################################

# Joining the Bankruptcy Data

bankruptcy = pd.read_excel(snakemake.input[1])


bdf = bankruptcy[["Cusip6", "Cusip9", "DateFiled", "DateEmerging", "DateRefile"]]

bdf['cusip'] = bdf['Cusip6'].astype(str) + bdf['Cusip9'].astype(str)
bdf['DateFiled'] = pd.to_datetime(bdf['DateFiled'], errors='coerce')
bdf['file fyearq'] = bdf['DateFiled'].dt.year
bdf['month'] = bdf['DateFiled'].dt.month
bdf['file fqtr'] = pd.cut(
    bdf['month'],
    bins=[0, 3, 6, 9, 12],
    labels=[1, 2, 3, 4]
).astype(int)
bdf = bdf.drop(columns=['month'])

bdf['DateEmerging'] = pd.to_datetime(bdf['DateEmerging'], errors='coerce')
bdf['emerge fyearq'] = bdf['DateEmerging'].dt.year
bdf['month'] = bdf['DateEmerging'].dt.month
bdf['emerge fqtr'] = pd.cut(
    bdf['month'],
    bins=[0, 3, 6, 9, 12],
    labels=[1, 2, 3, 4]
).astype('Int64')
bdf = bdf.drop(columns=['month'])

bdf['DateRefile'] = pd.to_datetime(bdf['DateRefile'], errors='coerce')
bdf['refile fyearq'] = bdf['DateRefile'].dt.year
bdf['month'] = bdf['DateRefile'].dt.month
bdf['refile fqtr'] = pd.cut(
    bdf['month'],
    bins=[0, 3, 6, 9, 12],
    labels=[1, 2, 3, 4]
).astype('Int64')
bdf = bdf.drop(columns=['month'])

bdf = bdf.drop(columns=["Cusip6", "Cusip9", "DateFiled", "DateEmerging", "DateRefile"])
bdf = bdf.drop_duplicates(subset='cusip', keep='first')
bdf = bdf[bdf["file fyearq"] >= 2006] # Keeping for only from 2006

# Identlfying common cusips

df1_cusips = df1.reset_index()["cusip"].unique().tolist()
bdf_cusips = bdf["cusip"].unique().tolist()
common_cusips = set(df1_cusips) & set(bdf_cusips)

# Ensuring all columns are in numeric format

df1 = df1.reset_index()
df1['fyearq'] = df1['fyearq'].astype(int)
df1['fqtr'] = df1['fqtr'].astype(int)
bdf['file fyearq'] = bdf['file fyearq'].astype(int)
bdf['file fqtr'] = bdf['file fqtr'].astype(int)
if 'emerge fyearq' in bdf.columns:
    bdf['emerge fyearq'] = pd.to_numeric(bdf['emerge fyearq'], errors='coerce')
if 'emerge fqtr' in bdf.columns:
    bdf['emerge fqtr'] = pd.to_numeric(bdf['emerge fqtr'], errors='coerce')
if 'refile fyearq' in bdf.columns:
    bdf['refile fyearq'] = pd.to_numeric(bdf['refile fyearq'], errors='coerce')
if 'refile fqtr' in bdf.columns:
    bdf['refile fqtr'] = pd.to_numeric(bdf['refile fqtr'], errors='coerce')

# Initiating the column

df1['default_status'] = 0

# Adding the default status column to the master df

for cusip in common_cusips:

    events = bdf[bdf['cusip'] == cusip]

    for _, event in events.iterrows():

        file_year, file_q = int(event['file fyearq']), int(event['file fqtr'])
        
        emerge_year = event.get('emerge fyearq', None)
        emerge_q = event.get('emerge fqtr', None)
        refile_year = event.get('refile fyearq', None)
        refile_q = event.get('refile fqtr', None)

        mask_bankruptcy = (
            (df1['cusip'] == cusip) &
            (
                ((df1['fyearq'] > file_year) |
                 ((df1['fyearq'] == file_year) & (df1['fqtr'] >= file_q)))
            )
        )

        if not pd.isna(emerge_year) and not pd.isna(emerge_q):
            mask_end_emerge = (
                (df1['cusip'] == cusip) &
                (
                    (df1['fyearq'] < emerge_year) |
                    ((df1['fyearq'] == emerge_year) & (df1['fqtr'] <= emerge_q))
                )
            )
            mask_D = mask_bankruptcy & mask_end_emerge
        else:
            mask_D = mask_bankruptcy

        df1.loc[mask_D, 'default_status'] = 'D'

        if not pd.isna(emerge_year) and not pd.isna(emerge_q):
            mask_after_emerge = (
                (df1['cusip'] == cusip) &
                (
                    (df1['fyearq'] > emerge_year) |
                    ((df1['fyearq'] == emerge_year) & (df1['fqtr'] > emerge_q))
                )
            )
            df1.loc[mask_after_emerge, 'default_status'] = 0

        if not pd.isna(refile_year) and not pd.isna(refile_q):
            mask_refile = (
                (df1['cusip'] == cusip) &
                (
                    ((df1['fyearq'] > refile_year) |
                     ((df1['fyearq'] == refile_year) & (df1['fqtr'] >= refile_q)))
                )
            )
            df1.loc[mask_refile, 'default_status'] = 'D'

df1['default_status'] = df1['default_status'].astype(str)

# Marking the defaulted observations as LIQ - Different from D in spcsrc. D is quality rating, poor performance< LIQ means filed for bankruptcy

df1 = df1[df1['spcsrc'] != 'LIQ']

df1.loc[df1['default_status'] == 'D', 'spcsrc'] = 'LIQ'

# df1[df1['spcsrc'] == 'LIQ'].sort_values(['fyearq', 'fqtr'])

df1 = df1.drop(columns = ["default_status"])

df1 = df1.set_index(["cusip", "fyearq", "fqtr"])

##################################################################################################################################

# Labelling LIQ as 1, others as 0

df1["default"] = df1["spcsrc"].isin(["LIQ"]).astype(int)

# df1.describe().T

# Making rf constant for each year

df1["rf"] = df1.groupby("fyearq")["optrfrq"].transform(
    lambda x: x.mean()
)

# df1.describe().T

##################################################################################################################################

# Creating the Quarterly Returns

retvol = pd.read_csv(snakemake.input[2])
df_for_merge = df1.copy().reset_index()

df_merged = df_for_merge.merge(
    retvol[['cusip', 'fyearq', 'fqtr', 'returns', 'volatility']],
    on=['cusip', 'fyearq', 'fqtr'],
    how='left'
)

# df1.describe().T
# df_merged.describe().T

df1 = df_merged.set_index(["cusip", "fyearq", "fqtr"])

df1["mktval"] = df1["prccq"]*df1["cshoq"]

df1 = df1.drop(columns=["mkvaltq", "optrfrq", "optvolq"])

# df1.reset_index().loc[df1.reset_index()["default"] == 1, "cusip"].nunique()
# df1["default"].sum()
# df1.reset_index()[["cusip"]].nunique()
# df1.describe().T

# Creating Returns

df1 = df1.sort_index(level=['cusip', 'fyearq', 'fqtr'])

# Removing non numeric for inputation

spcsrc = df1.reset_index()[["cusip", "fyearq", "fqtr", "spcsrc"]]
df1 = df1.drop(columns = ["spcsrc"])

# df1.reset_index().loc[df1.reset_index()["default"] == 1, "cusip"].nunique()
# df1["default"].sum()
# df1.reset_index()[["cusip"]].nunique()
# df1.describe().T

##################################################################################################################################

df1 = df1.reset_index()

# Imputing remaining values via iterative SVD Imputation

df_filled = iterative_svd_impute(df1)

# df_filled["cusip"].nunique() 2326
# df_filled.loc[df_filled["default"] == 1, "cusip"].nunique() 171
# df_filled["default"].sum() 4206
# len(df_filled) 150023

# Creating required columns for merton model

df_final = create_reqd_colums(data = df_filled, k=0.1) # Setting k = 0.1 following Z.Afik et al

# Merging the non numeric rating columns

df_final = pd.merge(df_final, spcsrc, on=["cusip", "fyearq", "fqtr"], how="left") 

##################################################################################################################################

# Implimenting the Merton Model of PD Estimation

df_merton = two_system_merton(df =df_final, E= "mktval", sigma_E="volatility", D="debt_level", rf = "rf", mu_a = "eqrt")

# Evaluating Model Performance

auc,pauc = compute_auc_pauc(df_merton.dropna()["default"], df_merton.dropna()["PD"], fpr_threshold=0.1)

df_merton = df_merton.reset_index()

df_merton.to_csv(snakemake.output[0], index = False)


model_performance = pd.DataFrame({
    'Head': [
        'Model Specification',
        'AUC',
        'PAUC'
    ],
    'Value': [
        '2sM-k:0.1-pauc:0.1',
        auc,
        pauc
    ]
})

model_performance.to_csv(snakemake.output[1], index = False)


#########################################################################################################################################

# # Plotting PD for each quality rating: Provides confidence in PD estimation - Idea: A+(has minimum), LIQ(has max), D(has second max)

# mean_pd = df_merton.groupby('spcsrc')['PD'].mean().sort_values(ascending=False)

# # Plot
# plt.figure(figsize=(10,5))
# mean_pd.plot(kind = 'bar')
# plt.xlabel('S&P Rating Category (spcsrc)')
# plt.ylabel('Mean Probability of Default (PD)')
# plt.title('Mean PD by S&P Rating Category')
# plt.grid(axis='y', alpha=0.3)
# plt.tight_layout()
# plt.show()



# # Group by both year and quarter
# pd_by_yq = df_merton.groupby(['fyearq', 'fqtr'])['sigma_A'].mean().reset_index()

# # Create a combined time label for plotting
# pd_by_yq['year_qtr'] = pd_by_yq['fyearq'].astype(str) + ' Q' + pd_by_yq['fqtr'].astype(str)

# plt.figure(figsize=(16,6))
# plt.plot(pd_by_yq['year_qtr'], pd_by_yq['sigma_A'], marker='o')
# plt.xlabel('Fiscal Year and Quarter')
# plt.ylabel('Mean Probability of Default (PD)')
# plt.title('Evolution of Mean PD by Year and Quarter')
# plt.xticks(rotation=90)
# plt.grid(True)
# plt.tight_layout()
# plt.show()