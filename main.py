import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import statsmodels.formula.api as smf

df = pd.read_csv('r_data.csv')

base = -1
means = {}

for i in range(0, 100, 1):
    df_i = df[df['Autocorrelation'] <= 0.1 * (i / 10) + base]
    if pd.notna(df_i['MKT'].mean()):
        key = i
        values = []
        values.append(base + 0.1 * (i / 10))
        values.append(df_i['MKT'].count())
        values.append(df_i['MKT'].mean())
        values.append(df_i['SMB'].mean())
        values.append(df_i['HML'].mean())
        means[key] = values

cumm_averages = pd.DataFrame.from_dict(means, orient='index',
                                       columns=['Autocorrelation Cut-off', 'Count', 'MKT Average', 'SMB Average',
                                                'HML Average'])

cumm_averages.plot(x='Autocorrelation Cut-off', y=['MKT Average', 'SMB Average', 'HML Average'],
                   color=['#222b35', '#8397b0', '#d6dce4'])
cumm_averages.plot(x='Autocorrelation Cut-off', y=['Count'], color=['#222b35', '#8397b0', '#d6dce4'])
plt.show()

cumm_averages.to_csv(r'cumm_averages_2.csv')

start = -0.27
stop = -0.15
ps = []
rs = []
bases = []
roofs = []
ks = []

ps_2 = []
rs_2 = []
bases_2 = []
roofs_2 = []
ks_2 = []

ps_3 = []
rs_3 = []
bases_3 = []
roofs_3 = []
ks_3 = []
for i in range(0, 100):
    base = start + i * 0.0006
    for j in range(0, 100):
        roof = stop - (j * 0.0006)
        df_j = df[df['Autocorrelation'] >= base]
        df_j_2 = df_j[df_j['Autocorrelation'] <= roof] 

        for k in range(0, 5):
            df_j_2_shifted = df_j_2.copy()
            try:
                df_j_2_shifted['MKT'] = df_j_2_shifted.MKT.shift(k)
                df_j_2_shifted['SMB'] = df_j_2_shifted.SMB.shift(k)
                df_j_2_shifted['HML'] = df_j_2_shifted.HML.shift(k)
                df_j_2_shifted = df_j_2_shifted.dropna(how='any', axis=0)
            except ValueError:
                break

            reg = 'HML~Autocorrelation'
            reg2 = 'SMB~Autocorrelation'
            reg3 = 'MKT~Autocorrelation'

            try:
                reg_res = smf.ols(reg, df_j_2_shifted).fit()
                reg_res2 = smf.ols(reg2, df_j_2_shifted).fit()
                reg_res3 = smf.ols(reg3, df_j_2_shifted).fit()
                p_value = reg_res.pvalues[1]
                p_value2 = reg_res2.pvalues[1]
                p_value3 = reg_res3.pvalues[1]
                r_value = reg_res.rsquared
                r_value_2 = reg_res2.rsquared
                r_value_3 = reg_res3.rsquared

                if p_value <= 0.05:
                    ps.append(p_value)
                    bases.append(base)
                    roofs.append(roof)
                    ks.append(k)
                    rs.append(r_value)
                if p_value2 <= 0.05:
                    ps_2.append(p_value2)
                    bases_2.append(base)
                    roofs_2.append(roof)
                    ks_2.append(k)
                    rs_2.append(r_value_2)
                if p_value3 <= 0.05:
                    ps_3.append(p_value3)
                    bases_3.append(base)
                    roofs_3.append(roof)
                    ks_3.append(k)
                    rs_3.append(r_value_3)

            except ValueError:
                break

res_hml = pd.DataFrame(list(zip(bases, roofs, ks, rs)), columns=['Base', 'Roof', 'Lag', 'Rsq'])
res_smb = pd.DataFrame(list(zip(bases_2, roofs_2, ks_2, rs_2)), columns=['Base', 'Roof', 'Lag', 'Rsq'])
res_mkt = pd.DataFrame(list(zip(bases_3, roofs_3, ks_3, rs_3)), columns=['Base', 'Roof', 'Lag', 'Rsq'])

df_treshold_one_MKT = df[df['Autocorrelation'] >= bases_3[-1]]
df_treshold_one_MKT_2 = df_treshold_one_MKT[df_treshold_one_MKT['Autocorrelation'] <= roofs_3[-1]]

df_treshold_one_SMB = df[df['Autocorrelation'] >= bases_2[-1]]
df_treshold_one_SMB_2 = df_treshold_one_SMB[df_treshold_one_SMB['Autocorrelation'] <= roofs_2[-1]]

df_treshold_one_HML = df[df['Autocorrelation'] >= bases[-1]]
df_treshold_one_HML_2 = df_treshold_one_HML[df_treshold_one_HML['Autocorrelation'] <= roofs[-1]]

res_hml.to_csv(r'res_hml_treshold_neg_1.csv')
res_smb.to_csv(r'res_smb_treshold_neg_1.csv')
res_mkt.to_csv(r'res_mkt_treshold_neg_1.csv')


#df_treshold_one_MKT_2.to_csv(r'treshold_one_MKT.csv')
#df_treshold_one_SMB_2.to_csv(r'treshold_one_SMB.csv')
#df_treshold_one_HML_2.to_csv(r'treshold_one_HML.csv')
#cumm_averages.to_csv(r'cumm_averages.csv')

test = df[df['Autocorrelation'] >= 0.2055]
test_2 = test[test['Autocorrelation'] <= 0.226499999999999]
test_2.to_csv(r'test_neg.csv')