import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv("Data\Prepared Data\prepared_data_regression.csv")
df = df.sort_values(['cusip', 'fyearq'])

#####################################################################################################################################

df['PD_lag'] = df.groupby('cusip')['PD'].shift(1)

df_model = df.dropna(subset=['PD', 'PD_lag'])

#####################################################################################################################################

model_PD = smf.ols('PD ~ PD_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_PD.summary()

df_model['residuals_PD'] = model_PD.resid

#####################################################################################################################################

model_news = smf.ols('net_bad_minus_good_news ~ PD_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_news.summary()

df_model['residuals_news'] = model_news.resid

#####################################################################################################################################

model_trial = smf.ols('residuals_PD ~ residuals_news', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_trial.summary()

#####################################################################################################################################

df_resid = df_model[['cusip', 'gics4', 'residuals_PD', 'residuals_news', 'average_risk']].dropna()

q1 = df_resid['average_risk'].quantile(0.25)
q2 = df_resid['average_risk'].quantile(0.50)
q3 = df_resid['average_risk'].quantile(0.75)

# Assign risk based on quartiles
df_resid['risk'] = (df_resid['average_risk'] >= q2).astype(int)

model_trial_with_risk = smf.ols('residuals_PD ~ residuals_news + risk', data=df_resid).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_resid['cusip']}
)

model_trial_with_risk.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 0]

model_no_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_no_risk_cusip.summary()

model_no_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)

model_no_risk_gics.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 1]

model_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_risk_cusip.summary()

model_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)

model_risk_gics.summary()






###################################################################################################################################




import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv("Data\Prepared Data\prepared_data_regression.csv")
df = df.sort_values(['cusip', 'fyearq'])

#####################################################################################################################################

df['log_vol'] = np.log(df['volatility'])
df['log_vol_lag'] = df.groupby('cusip')['log_vol'].shift(1)

df_model = df.dropna(subset=['log_vol', 'log_vol_lag'])

#####################################################################################################################################

model_PD = smf.ols('log_vol ~ log_vol_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_PD.summary()

df_model['residuals_PD'] = model_PD.resid

#####################################################################################################################################

model_news = smf.ols('net_bad_minus_good_news ~ log_vol_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_news.summary()

df_model['residuals_news'] = model_news.resid

#####################################################################################################################################

model_trial = smf.ols('residuals_PD ~ residuals_news', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_trial.summary()

#####################################################################################################################################

df_resid = df_model[['cusip', 'gics4', 'residuals_PD', 'residuals_news', 'average_risk']].dropna()

q1 = df_resid['average_risk'].quantile(0.25)
q2 = df_resid['average_risk'].quantile(0.50)
q3 = df_resid['average_risk'].quantile(0.75)

# Assign risk based on quartiles
df_resid['risk'] = (df_resid['average_risk'] >= q2).astype(int)

model_trial_with_risk = smf.ols('residuals_PD ~ residuals_news + risk', data=df_resid).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_resid['cusip']}
)

model_trial_with_risk.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 0]

model_no_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_no_risk_cusip.summary()

model_no_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)

model_no_risk_gics.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 1]

model_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_risk_cusip.summary()





model_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)

model_risk_gics.summary()


#####################################################################################################################################




import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv("Data\Prepared Data\prepared_data_regression.csv")
df = df.sort_values(['cusip', 'fyearq'])

#####################################################################################################################################

df['invrmq_lag'] = df.groupby('cusip')['invrmq'].shift(1)

df_model = df.dropna(subset=['invrmq', 'invrmq_lag'])

#####################################################################################################################################

model_PD = smf.ols('invrmq ~ invrmq_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_PD.summary()

df_model['residuals_PD'] = model_PD.resid

#####################################################################################################################################

model_news = smf.ols('net_bad_minus_good_news ~ invrmq_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_news.summary()

df_model['residuals_news'] = model_news.resid

#####################################################################################################################################

model_trial = smf.ols('residuals_PD ~ residuals_news', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_trial.summary()

#####################################################################################################################################

df_resid = df_model[['cusip', 'gics4', 'residuals_PD', 'residuals_news', 'average_risk']].dropna()

q1 = df_resid['average_risk'].quantile(0.25)
q2 = df_resid['average_risk'].quantile(0.50)
q3 = df_resid['average_risk'].quantile(0.75)

# Assign risk based on quartiles
df_resid['risk'] = (df_resid['average_risk'] >= q2).astype(int)

model_trial_with_risk = smf.ols('residuals_PD ~ residuals_news + risk', data=df_resid).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_resid['cusip']}
)

model_trial_with_risk.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 0]

model_no_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_no_risk_cusip.summary()

model_no_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)

model_no_risk_gics.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 1]

model_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_risk_cusip.summary()

model_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)


model_risk_gics.summary()






#####################################################################################################################################




import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv("Data\Prepared Data\prepared_data_regression.csv")
df = df.sort_values(['cusip', 'fyearq'])

#####################################################################################################################################

df['cogsq_lag'] = df.groupby('cusip')['cogsq'].shift(1)

df_model = df.dropna(subset=['cogsq', 'cogsq_lag'])

#####################################################################################################################################

model_PD = smf.ols('cogsq ~ cogsq_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_PD.summary()

df_model['residuals_PD'] = model_PD.resid

#####################################################################################################################################

model_news = smf.ols('net_bad_minus_good_news ~ cogsq_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_news.summary()

df_model['residuals_news'] = model_news.resid

#####################################################################################################################################

model_trial = smf.ols('residuals_PD ~ residuals_news', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_trial.summary()

#####################################################################################################################################

df_resid = df_model[['cusip', 'gics4', 'residuals_PD', 'residuals_news', 'average_risk']].dropna()

q1 = df_resid['average_risk'].quantile(0.25)
q2 = df_resid['average_risk'].quantile(0.50)
q3 = df_resid['average_risk'].quantile(0.75)

# Assign risk based on quartiles
df_resid['risk'] = (df_resid['average_risk'] >= q2).astype(int)

model_trial_with_risk = smf.ols('residuals_PD ~ residuals_news + risk', data=df_resid).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_resid['cusip']}
)

model_trial_with_risk.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 0]

model_no_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_no_risk_cusip.summary()

model_no_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)

model_no_risk_gics.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 1]

model_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_risk_cusip.summary()

model_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)


model_risk_gics.summary()






#####################################################################################################################################




import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv("Data\Prepared Data\prepared_data_regression.csv")
df = df.sort_values(['cusip', 'fyearq'])

#####################################################################################################################################

df['invtq_lag'] = df.groupby('cusip')['invtq'].shift(1)

df_model = df.dropna(subset=['invtq', 'invtq_lag'])

#####################################################################################################################################

model_PD = smf.ols('invtq ~ invtq_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_PD.summary()

df_model['residuals_PD'] = model_PD.resid

#####################################################################################################################################

model_news = smf.ols('net_bad_minus_good_news ~ invtq_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_news.summary()

df_model['residuals_news'] = model_news.resid

#####################################################################################################################################

model_trial = smf.ols('residuals_PD ~ residuals_news', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_trial.summary()

#####################################################################################################################################

df_resid = df_model[['cusip', 'gics4', 'residuals_PD', 'residuals_news', 'average_risk']].dropna()

q1 = df_resid['average_risk'].quantile(0.25)
q2 = df_resid['average_risk'].quantile(0.50)
q3 = df_resid['average_risk'].quantile(0.75)

# Assign risk based on quartiles
df_resid['risk'] = (df_resid['average_risk'] >= q2).astype(int)

model_trial_with_risk = smf.ols('residuals_PD ~ residuals_news + risk', data=df_resid).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_resid['cusip']}
)

model_trial_with_risk.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 0]

model_no_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_no_risk_cusip.summary()

model_no_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)

model_no_risk_gics.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 1]

model_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_risk_cusip.summary()

model_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)


model_risk_gics.summary()






#####################################################################################################################################




import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv("Data\Prepared Data\prepared_data_regression.csv")
df = df.sort_values(['cusip', 'fyearq'])

#####################################################################################################################################

df['invo'] = df['invtq'] - df['invrmq']

df['invo_lag'] = df.groupby('cusip')['invo'].shift(1)

df_model = df.dropna(subset=['invo', 'invo_lag'])

#####################################################################################################################################

model_PD = smf.ols('invo ~ invo_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_PD.summary()

df_model['residuals_PD'] = model_PD.resid

#####################################################################################################################################

model_news = smf.ols('net_bad_minus_good_news ~ invo_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_news.summary()

df_model['residuals_news'] = model_news.resid

#####################################################################################################################################

model_trial = smf.ols('residuals_PD ~ residuals_news', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_trial.summary()

#####################################################################################################################################

df_resid = df_model[['cusip', 'gics4', 'residuals_PD', 'residuals_news', 'average_risk']].dropna()

q1 = df_resid['average_risk'].quantile(0.25)
q2 = df_resid['average_risk'].quantile(0.50)
q3 = df_resid['average_risk'].quantile(0.75)

# Assign risk based on quartiles
df_resid['risk'] = (df_resid['average_risk'] >= q2).astype(int)

model_trial_with_risk = smf.ols('residuals_PD ~ residuals_news + risk', data=df_resid).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_resid['cusip']}
)

model_trial_with_risk.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 0]

model_no_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_no_risk_cusip.summary()

model_no_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)

model_no_risk_gics.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 1]

model_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_risk_cusip.summary()

model_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)


model_risk_gics.summary()









#####################################################################################################################################




import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv("Data\Prepared Data\prepared_data_regression.csv")
df = df.sort_values(['cusip', 'fyearq'])

#####################################################################################################################################

df['gdwlq_lag'] = df.groupby('cusip')['gdwlq'].shift(1)

df_model = df.dropna(subset=['gdwlq', 'gdwlq_lag'])

#####################################################################################################################################

model_PD = smf.ols('gdwlq ~ gdwlq_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_PD.summary()

df_model['residuals_PD'] = model_PD.resid

#####################################################################################################################################

model_news = smf.ols('net_bad_minus_good_news ~ gdwlq_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_news.summary()

df_model['residuals_news'] = model_news.resid

#####################################################################################################################################

model_trial = smf.ols('residuals_PD ~ residuals_news', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_trial.summary()

#####################################################################################################################################

df_resid = df_model[['cusip', 'gics4', 'residuals_PD', 'residuals_news', 'average_risk']].dropna()

q1 = df_resid['average_risk'].quantile(0.25)
q2 = df_resid['average_risk'].quantile(0.50)
q3 = df_resid['average_risk'].quantile(0.75)

# Assign risk based on quartiles
df_resid['risk'] = (df_resid['average_risk'] >= q2).astype(int)

model_trial_with_risk = smf.ols('residuals_PD ~ residuals_news + risk', data=df_resid).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_resid['cusip']}
)

model_trial_with_risk.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 0]

model_no_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_no_risk_cusip.summary()

model_no_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)

model_no_risk_gics.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 1]

model_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_risk_cusip.summary()

model_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)


model_risk_gics.summary()






#####################################################################################################################################




import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv("Data\Prepared Data\prepared_data_regression.csv")
df = df.sort_values(['cusip', 'fyearq'])

#####################################################################################################################################

df['short_term_debt_lag'] = df.groupby('cusip')['short_term_debt'].shift(1)

df_model = df.dropna(subset=['short_term_debt', 'short_term_debt_lag'])

#####################################################################################################################################

model_PD = smf.ols('short_term_debt ~ short_term_debt_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_PD.summary()

df_model['residuals_PD'] = model_PD.resid

#####################################################################################################################################

model_news = smf.ols('net_bad_minus_good_news ~ short_term_debt_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_news.summary()

df_model['residuals_news'] = model_news.resid

#####################################################################################################################################

model_trial = smf.ols('residuals_PD ~ residuals_news', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_trial.summary()

#####################################################################################################################################

df_resid = df_model[['cusip', 'gics4', 'residuals_PD', 'residuals_news', 'average_risk']].dropna()

q1 = df_resid['average_risk'].quantile(0.25)
q2 = df_resid['average_risk'].quantile(0.50)
q3 = df_resid['average_risk'].quantile(0.75)

# Assign risk based on quartiles
df_resid['risk'] = (df_resid['average_risk'] >= q2).astype(int)

model_trial_with_risk = smf.ols('residuals_PD ~ residuals_news + risk', data=df_resid).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_resid['cusip']}
)

model_trial_with_risk.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 0]

model_no_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_no_risk_cusip.summary()

model_no_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)

model_no_risk_gics.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 1]

model_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_risk_cusip.summary()

model_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)


model_risk_gics.summary()






#####################################################################################################################################




import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv("Data\Prepared Data\prepared_data_regression.csv")
df = df.sort_values(['cusip', 'fyearq'])

#####################################################################################################################################

df['eqrt_lag'] = df.groupby('cusip')['eqrt'].shift(1)

df_model = df.dropna(subset=['eqrt', 'eqrt_lag'])

#####################################################################################################################################

model_PD = smf.ols('eqrt ~ eqrt_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_PD.summary()

df_model['residuals_PD'] = model_PD.resid

#####################################################################################################################################

model_news = smf.ols('net_bad_minus_good_news ~ eqrt_lag + C(fyearq) + C(spcsrc) + C(gics4) + negative', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_news.summary()

df_model['residuals_news'] = model_news.resid

#####################################################################################################################################

model_trial = smf.ols('residuals_PD ~ residuals_news', data=df_model).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_model['cusip']}
)

model_trial.summary()

#####################################################################################################################################

df_resid = df_model[['cusip', 'gics4', 'residuals_PD', 'residuals_news', 'average_risk']].dropna()

q1 = df_resid['average_risk'].quantile(0.25)
q2 = df_resid['average_risk'].quantile(0.50)
q3 = df_resid['average_risk'].quantile(0.75)

# Assign risk based on quartiles
df_resid['risk'] = (df_resid['average_risk'] >= q2).astype(int)

model_trial_with_risk = smf.ols('residuals_PD ~ residuals_news + risk', data=df_resid).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_resid['cusip']}
)

model_trial_with_risk.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 0]

model_no_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_no_risk_cusip.summary()

model_no_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)

model_no_risk_gics.summary()

#####################################################################################################################################

df_subset = df_resid[df_resid['risk'] == 1]

model_risk_cusip = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['cusip']}
)

model_risk_cusip.summary()

model_risk_gics = smf.ols('residuals_PD ~ residuals_news', data=df_subset).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_subset['gics4']}
)


model_risk_gics.summary()
