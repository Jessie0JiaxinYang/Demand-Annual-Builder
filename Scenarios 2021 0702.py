# -*- coding: utf-8 -*-
# 2021 0428 
# finished com, ind, res, trn in one file


#  2021 0625 transfer the format that Birdy needs.
# 2021 0628 : big bug.....
# line 125, data should equal to "country_data", not "data". If just "data", one method under different 
# countries are the same, see in "scenarios-results total 2021 0624 and you will find that.

# 2021 0628: add forecast into this 

# 2021 0701: delete when independent variable is more than 5

# 2021 0702: delete wrong combination which has cross-term variables
#            add pet sector into this 


import pandas as pd
import numpy as np
from itertools import combinations
import datetime
from pandas.core.frame import DataFrame
import math

starttime = datetime.datetime.now()

import statsmodels.formula.api as smf  # this allows us to write down our statistical model with an R-like string
import statsmodels.stats.api as sms  # not needed unless you want to run a test
import statsmodels.api as sm # not needed unless you want to do something fancier
from statsmodels.compat import lzip # useful for printing out complicated test statistics, also not needed now

table1 = pd.read_excel(r'C:\Users\jiaxi\OneDrive\桌面\RBAC 2021\Demand Annual Builder\Input Data 2021 0702-run.xlsx', sheet_name = 'Target2')
table =table1.iloc[6:, 29: ]

table.columns = [ 'Country', 'Year', 'lcom', 'lelc', 'lind', 'lres', 'ltrn', 'lpet' ,'lgdp', 'lppl', 'lgdpp', 'lgdp_man', 'lche', 'lfoo', 'lmac', 'ltex', 'loth', 'luppl', 'llf', 'lgop', 'lgou', 'lpou', 'lgasprice', 'loilprice']
table[['Year']] = table[['Year']].astype('float')
table[['lcom']] = table[['lcom']].astype('float')
table[['lelc']] = table[['lelc']].astype('float')
table[['lind']] = table[['lind']].astype('float')
table[['lres']] = table[['lres']].astype('float')
table[['ltrn']] = table[['ltrn']].astype('float')
table[['lpet']] = table[['lpet']].astype('float')
table[['lgdp']] = table[['lgdp']].astype('float')
table[['lppl']] = table[['lppl']].astype('float')
table[['lgdpp']] = table[['lgdpp']].astype('float')
table[['lgdp_man']] = table[['lgdp_man']].astype('float')
table[['lche']] = table[['lche']].astype('float')
table[['lfoo']] = table[['lfoo']].astype('float')
table[['lmac']] = table[['lmac']].astype('float')
table[['ltex']] = table[['ltex']].astype('float')
table[['loth']] = table[['loth']].astype('float')
table[['luppl']] = table[['luppl']].astype('float')
table[['llf']] = table[['llf']].astype('float')
table[['lgop']] = table[['lgop']].astype('float')
table[['lgou']] = table[['lgou']].astype('float')
table[['lpou']] = table[['lpou']].astype('float')
table[['lgasprice']] = table[['lgasprice']].astype('float')
table[['loilprice']] = table[['loilprice']].astype('float')

yrs = pd.read_excel(r'C:\Users\jiaxi\OneDrive\桌面\RBAC 2021\Demand Annual Builder\Input Final Scenario.xlsx', sheet_name = 'Yr_Index')

# COM
data = table.drop(['lelc', 'lind', 'lres', 'ltrn',  'lgdp_man', 'lche', 'lfoo', 'lmac', 'ltex', 'loth', 'lgasprice', 'loilprice'], axis = 1)

r_table = pd.read_excel(r'C:\Users\jiaxi\OneDrive\桌面\RBAC 2021\Demand Annual Builder\Input Scenarios.xlsx', sheet_name = 'COM')
comcountry = r_table.loc[:,['Country ID',  'Country']]

country_name = r_table['Country'].to_list()

ref_country = table['Country'].unique()

df = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()


for i in country_name:
    if i != 'ALL':
        if i in ref_country:

                Country = data[data['Country']== i]
                country_filter = r_table[r_table['Country']== i]
                
                retr_country = country_filter['Country'].to_list()[0]
                
                country_data = data[data['Country']==retr_country]

                x_labels = country_filter[['x0','x1','x2','x3','x4','x5','x6','x7', 'x8', 'x9',  'y']]
                rid = country_filter['Country ID'].tolist()[0]
                y_var = x_labels['y'].to_list()[0]

                def subcombs_2(dset):
                    data_ = []
                    for i in range(1,len(dset)+1):
                        for j in combinations(dset,i):
                            data_.append(list(j))
                    return data_


                def stitch(f):
                    out = ""
                    for k,i in enumerate(f):
                        if len(i) > 1:
                            out += i
                        else:
                            try:
                                out += i
                            except:
                                out +=i[0]
                        if k < len(f)-1:
                            out+=" + "
                    return out

                regressors = ['lgdp', 'lppl', 'lgdpp', 'luppl', 'llf']
                allcombs = subcombs_2(regressors)               
                del allcombs[30:]
                allcombs.append(['lgdp', 'lppl', 'lgop'])
                allcombs.append(['lgdp', 'luppl', 'lgou'])
                allcombs.append(['lppl', 'luppl', 'lpou'])
                
                aclist = []

                for i in allcombs:
                     aclist.append('lcom~'+stitch(i)) 

                for i in aclist:
                    model = smf.ols(i, data=country_data)
                    results = model.fit()
                    a = i.replace('lcom~', '')
                    
                    values = pd.DataFrame(results.params).reset_index()
                    values.columns = ['names', 'coef']
                    xfinal = x_labels
                    xfinal['Intercept'] = 'Intercept'
                    xfinal = xfinal.T.reset_index()
                    xfinal.columns = ['labels', 'names']
                    export_val = xfinal.merge(values, on = 'names', how='left')
                    results.tvalues
                    t_stat = pd.DataFrame(results.tvalues).reset_index()
                    t_stat.columns = ['names', 't-statistic']
                    export_val1 = export_val.merge(t_stat, on = 'names', how='left')
                    
                    
                    r_columns = ['Year'] + values.xs("names", axis = 1).tolist()
                    del r_columns[1: 2]
                    forecast_var = country_data[r_columns].reset_index(drop=True)
                    forecast_var['Intercept'] = 1           
                    test = forecast_var[forecast_var['Year']>2016]
                    xval = test.drop(columns = ['Year'])

                    
                    entry2 = pd.DataFrame({'Country ID': [rid, rid, rid, rid,rid, rid, rid, rid, rid, rid, rid],
                                   'R-square': [results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared],
                                   'RegType': [a, a, a, a, a, a, a, a, a, a, a],
                                   'CoeID': ['intercept', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'],
                                   'Coe': [export_val1[export_val1['labels']=='Intercept']['coef'].to_list()[0], export_val1[export_val1['labels']=='x0']['coef'].to_list()[0], export_val1[export_val1['labels']=='x1']['coef'].to_list()[0], export_val1[export_val1['labels']=='x2']['coef'].to_list()[0], export_val1[export_val1['labels']=='x3']['coef'].to_list()[0], export_val1[export_val1['labels']=='x4']['coef'].to_list()[0], export_val1[export_val1['labels']=='x5']['coef'].to_list()[0], export_val1[export_val1['labels']=='x6']['coef'].to_list()[0], export_val1[export_val1['labels']=='x7']['coef'].to_list()[0], export_val1[export_val1['labels']=='x8']['coef'].to_list()[0], export_val1[export_val1['labels']=='x9']['coef'].to_list()[0]],
                                   'tvalueID': ['t_intercept', 't_x0', 't_x1', 't_x2', 't_x3', 't_x4', 't_x5', 't_x6', 't_x7', 't_x8', 't_x9'],
                                   'tvalue': [export_val1[export_val1['labels']=='Intercept']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x0']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x1']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x2']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x3']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x4']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x5']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x6']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x7']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x8']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x9']['t-statistic'].to_list()[0]]})
                    
                    y_hat = results.predict(xval) 
                    yr_p = yrs[yrs['Year']>=2017]            
                    p_export = pd.concat([yr_p, y_hat], axis=1, sort=False)
                    p_export['Country ID'] = rid
                    p_export["RegType"] = a
                    
                    cagr = p_export
                    cagr.columns = ['Year', 'lvalue', 'Country ID', 'RegType']
                    cagr['value'] = math.exp(1) ** cagr['lvalue']                    
                    cagr['test1'] = cagr['Year'] % 5
                    def fun(x):
                      if x == 2017:
                        return 0
                      else:
                        return 1
                    cagr['test2'] =cagr['Year'].apply(lambda x: fun(x))
                    cagr['test'] = cagr['test1'] * cagr['test2']
                    cagr = cagr.drop(cagr[cagr.test > 0].index)
                    cagr['CAGR'] = (cagr['value'] / cagr['value'].shift(+1) ) ** (1/(cagr['Year'] - cagr['Year'].shift(+1))) - 1
                    cagr = cagr.drop(['test1', 'test2', 'test'], axis = 1)
                    
                    p_export = p_export.drop(['test1', 'test2', 'test'], axis = 1)

                    df = pd.concat([df,entry2])
                    df2 = pd.concat([df2,p_export])
                    df3 = pd.concat([df3,cagr])
                    
            
df.insert(1, "Sector", "com")
df = df[df['Coe'].notna()]
df = pd.merge(df,comcountry,on='Country ID',how='inner') 

df2["Sector"] = 'com'
df2.columns = ['Year', 'lvalue', 'Country ID', 'RegType', 'Sector', 'value']             
df2 = pd.merge(df2,comcountry,on='Country ID',how='inner') 

df3["Sector"] = 'com'
df3 = pd.merge(df3,comcountry,on='Country ID',how='inner') 

allcombs2  = DataFrame(aclist)
allcombs2["ID"] =  range(len(allcombs2)) 
allcombs2["ID"] = allcombs2["ID"] + 1
allcombs2["Sector"] = "com"
allcombs2['RegID'] = allcombs2['ID'].apply(lambda x: 'com00' + str(x)  if x<10 else 'com0' +str(x) if x<100 else 'com' + str(x))
allcombs2.columns = ['RegType', 'ID', 'Sector', 'RegID']
allcombs2['RegType']  = allcombs2['RegType'].str.replace("lcom~", "")
com_regid = allcombs2.drop(['ID','Sector'],axis = 1)

com = pd.merge(df,com_regid,on='RegType',how='inner') 
com_p = pd.merge(df2,com_regid,on='RegType',how='inner') 
com_cagr = pd.merge(df3,com_regid,on='RegType',how='inner') 
com_regid["Sector"] = "com"

endtime = datetime.datetime.now()
print (endtime - starttime)

print("Congradulations! Com Finished :)")


# IND
data = table.drop(['lelc', 'ltrn', 'lcom', 'lres', 'lpet', 'lgasprice', 'loilprice'], axis = 1)

r_table = pd.read_excel(r'C:\Users\jiaxi\OneDrive\桌面\RBAC 2021\Demand Annual Builder\Input Scenarios.xlsx', sheet_name = 'IND')
indcountry = r_table.loc[:,['Country ID',  'Country']]

country_name = r_table['Country'].to_list()

ref_country = table['Country'].unique()

df = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()


for i in country_name:
    if i != 'ALL':
        if i in ref_country:

                Country = data[data['Country']== i]
                country_filter = r_table[r_table['Country']== i]
                
                retr_country = country_filter['Country'].to_list()[0]
                
                country_data = data[data['Country']==retr_country]

                x_labels = country_filter[['x0','x1','x2','x3','x4','x5','x6','x7', 'x8', 'x9',  'y']]
                rid = country_filter['Country ID'].tolist()[0]
                y_var = x_labels['y'].to_list()[0]

                def subcombs_2(dset):
                    data_ = []
                    for i in range(1,len(dset)+1):
                        for j in combinations(dset,i):
                            data_.append(list(j))
                    return data_


                def stitch(f):
                    out = ""
                    for k,i in enumerate(f):
                        if len(i) > 1:
                            out += i
                        else:
                            try:
                                out += i
                            except:
                                out +=i[0]
                        if k < len(f)-1:
                            out+=" + "
                    return out

                regressors = ['lgdp', 'lppl', 'lgdpp', 'luppl', 'llf', 'lgdp_man']
                
                allcombs = subcombs_2(regressors)
                del allcombs[56:]
                allcombs.append(['lgdp', 'lppl', 'lgop'])
                allcombs.append(['lgdp', 'luppl', 'lgou'])
                allcombs.append(['lppl', 'luppl', 'lpou'])
                
                aclist = []

                for i in allcombs:
                     aclist.append('lind~'+stitch(i)) 

                for i in aclist:
                    model = smf.ols(i, data=country_data)
                    results = model.fit()
                    a = i.replace('lind~', '')
                    
                    values = pd.DataFrame(results.params).reset_index()
                    values.columns = ['names', 'coef']
                    xfinal = x_labels
                    xfinal['Intercept'] = 'Intercept'
                    xfinal = xfinal.T.reset_index()
                    xfinal.columns = ['labels', 'names']
                    export_val = xfinal.merge(values, on = 'names', how='left')
                    results.tvalues
                    t_stat = pd.DataFrame(results.tvalues).reset_index()
                    t_stat.columns = ['names', 't-statistic']
                    export_val1 = export_val.merge(t_stat, on = 'names', how='left')
                    
                    
                    r_columns = ['Year'] + values.xs("names", axis = 1).tolist()
                    del r_columns[1: 2]
                    forecast_var = country_data[r_columns].reset_index(drop=True)
                    forecast_var['Intercept'] = 1           
                    test = forecast_var[forecast_var['Year']>2016]
                    xval = test.drop(columns = ['Year'])

                    
                    entry2 = pd.DataFrame({'Country ID': [rid, rid, rid, rid,rid, rid, rid, rid, rid, rid, rid],
                                   'R-square': [results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared],
                                   'RegType': [a, a, a, a, a, a, a, a, a, a, a],
                                   'CoeID': ['intercept', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'],
                                   'Coe': [export_val1[export_val1['labels']=='Intercept']['coef'].to_list()[0], export_val1[export_val1['labels']=='x0']['coef'].to_list()[0], export_val1[export_val1['labels']=='x1']['coef'].to_list()[0], export_val1[export_val1['labels']=='x2']['coef'].to_list()[0], export_val1[export_val1['labels']=='x3']['coef'].to_list()[0], export_val1[export_val1['labels']=='x4']['coef'].to_list()[0], export_val1[export_val1['labels']=='x5']['coef'].to_list()[0], export_val1[export_val1['labels']=='x6']['coef'].to_list()[0], export_val1[export_val1['labels']=='x7']['coef'].to_list()[0], export_val1[export_val1['labels']=='x8']['coef'].to_list()[0], export_val1[export_val1['labels']=='x9']['coef'].to_list()[0]],
                                   'tvalueID': ['t_intercept', 't_x0', 't_x1', 't_x2', 't_x3', 't_x4', 't_x5', 't_x6', 't_x7', 't_x8', 't_x9'],
                                   'tvalue': [export_val1[export_val1['labels']=='Intercept']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x0']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x1']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x2']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x3']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x4']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x5']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x6']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x7']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x8']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x9']['t-statistic'].to_list()[0]]})
                    
                    y_hat = results.predict(xval) 
                    yr_p = yrs[yrs['Year']>=2017]            
                    p_export = pd.concat([yr_p, y_hat], axis=1, sort=False)
                    p_export['Country ID'] = rid
                    p_export["RegType"] = a
                    
                    cagr = p_export
                    cagr.columns = ['Year', 'lvalue', 'Country ID', 'RegType']
                    cagr['value'] = math.exp(1) ** cagr['lvalue']                    
                    cagr['test1'] = cagr['Year'] % 5
                    def fun(x):
                      if x == 2017:
                        return 0
                      else:
                        return 1
                    cagr['test2'] =cagr['Year'].apply(lambda x: fun(x))
                    cagr['test'] = cagr['test1'] * cagr['test2']
                    cagr = cagr.drop(cagr[cagr.test > 0].index)
                    cagr['CAGR'] = (cagr['value'] / cagr['value'].shift(+1) ) ** (1/(cagr['Year'] - cagr['Year'].shift(+1))) - 1
                    cagr = cagr.drop(['test1', 'test2', 'test'], axis = 1)
                    
                    p_export = p_export.drop(['test1', 'test2', 'test'], axis = 1)

                    df = pd.concat([df,entry2])
                    df2 = pd.concat([df2,p_export])
                    df3 = pd.concat([df3,cagr])
                    
            
df.insert(1, "Sector", "ind")
df = df[df['Coe'].notna()]
df = pd.merge(df,indcountry,on='Country ID',how='inner') 

df2["Sector"] = 'ind'
df2.columns = ['Year', 'lvalue', 'Country ID', 'RegType', 'Sector', 'value']             
df2 = pd.merge(df2,indcountry,on='Country ID',how='inner') 

df3["Sector"] = 'ind'
df3 = pd.merge(df3,indcountry,on='Country ID',how='inner') 

allcombs2  = DataFrame(aclist)
allcombs2["ID"] =  range(len(allcombs2)) 
allcombs2["ID"] = allcombs2["ID"] + 1
allcombs2["Sector"] = "ind"
allcombs2['RegID'] = allcombs2['ID'].apply(lambda x: 'ind00' + str(x)  if x<10 else 'ind0' +str(x) if x<100 else 'ind' + str(x))
allcombs2.columns = ['RegType', 'ID', 'Sector', 'RegID']
allcombs2['RegType']  = allcombs2['RegType'].str.replace("lind~", "")
ind_regid = allcombs2.drop(['ID','Sector'],axis = 1)

ind = pd.merge(df,ind_regid,on='RegType',how='inner') 
ind_p = pd.merge(df2,ind_regid,on='RegType',how='inner')
ind_cagr = pd.merge(df3,ind_regid,on='RegType',how='inner') 


## IND2

main_data = table.drop(['lelc', 'lcom', 'lres', 'ltrn', 'lpet',  'lgdp_man', 'lgdp', 'lppl', 'lgdpp', 'lgdp_man',  'luppl', 'llf', 'lgop', 'lgou', 'lpou', 'lgasprice', 'loilprice'], axis = 1)

# main_data = pd.read_excel('/Users/adishluitel/Desktop/RBAC_Work/Trial/PY_Workflow.xlsx', sheet_name = 'Input_Data')
r_table = pd.read_excel(r'C:\Users\jiaxi\OneDrive\桌面\RBAC 2021\Demand Annual Builder\Input Scenarios.xlsx', sheet_name = 'IND2')
ind2country = r_table.loc[:,['Country ID',  'Country']]

country_name = r_table['Country'].to_list()
ref_country = main_data['Country'].unique()
df = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()


for i in country_name:
    if i != 'ALL':
        if i in ref_country:


            country_filter = r_table[r_table['Country']== i]

            retr_country = country_filter['Country'].to_list()[0]

            country_data = main_data[main_data['Country']==retr_country]

            x_labels = country_filter[['x0','x1','x2','x3','x4','x5','x6','x7','y']]
               
            x_labels2 =x_labels.astype(str)    
            a =       x_labels2.iloc[0,8] + " " + x_labels2.iloc[0,0] + " + " + x_labels2.iloc[0,1] + " + " + x_labels2.iloc[0,2] + \
                " + " + x_labels2.iloc[0,3] + " + "+ x_labels2.iloc[0,4] + " + "+ x_labels2.iloc[0,5] + \
                " + " + x_labels2.iloc[0,6] + " + "+ x_labels2.iloc[0,7] 
            a = a.replace(' + nan', '')
            a = a.replace('lind ', '')
    
            rid = country_filter['Country ID'].tolist()[0]
            y_var = x_labels['y'].to_list()[0]

            r_columns = ['Year'] + x_labels.dropna(axis=1,how='all').reset_index(drop=True).loc[0].tolist()

            regressors = country_data[r_columns].reset_index(drop=True)
            regressors['intercept'] = 1

            train = regressors[regressors['Year']<=2016]
            test = regressors[regressors['Year']>2016]

            X = train.drop(columns = ['Year',y_var]) 
            y = train[[y_var]]
            xval = test.drop(columns = ['Year',y_var])

            model = sm.OLS(y, X, missing='drop')
            results = model.fit()

            y_hat = results.predict(xval) 

            values = pd.DataFrame(results.params).reset_index()

            values.columns = ['names', 'coef']

            xfinal = x_labels
            xfinal['intercept'] = 'intercept'
            xfinal = xfinal.T.reset_index()

            xfinal.columns = ['labels', 'names']

            export_val = xfinal.merge(values, on = 'names', how='left')
            t_stat = pd.DataFrame(results.tvalues).reset_index()
            t_stat.columns = ['names', 't-statistic']

            export_val1 = export_val.merge(t_stat, on = 'names', how='left')

            entry2 = pd.DataFrame({'Country ID': [rid, rid, rid, rid,rid, rid, rid, rid, rid],
                                   'R-square': [results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared],
                                   'RegType': [a, a, a, a, a, a, a, a, a],
                                   'CoeID': ['intercept', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'],
                                   'Coe': [export_val1[export_val1['labels']=='intercept']['coef'].to_list()[0], export_val1[export_val1['labels']=='x0']['coef'].to_list()[0], export_val1[export_val1['labels']=='x1']['coef'].to_list()[0], export_val1[export_val1['labels']=='x2']['coef'].to_list()[0], export_val1[export_val1['labels']=='x3']['coef'].to_list()[0], export_val1[export_val1['labels']=='x4']['coef'].to_list()[0], export_val1[export_val1['labels']=='x5']['coef'].to_list()[0], export_val1[export_val1['labels']=='x6']['coef'].to_list()[0], export_val1[export_val1['labels']=='x7']['coef'].to_list()[0]],
                                   'tvalueID': ['t_intercept', 't_x0', 't_x1', 't_x2', 't_x3', 't_x4', 't_x5', 't_x6', 't_x7'],
                                   'tvalue': [export_val1[export_val1['labels']=='intercept']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x0']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x1']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x2']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x3']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x4']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x5']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x6']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x7']['t-statistic'].to_list()[0]]})
                                   
            y_hat = results.predict(xval) 
            yr_p = yrs[yrs['Year']>=2017]            
            p_export = pd.concat([yr_p, y_hat], axis=1, sort=False)
            p_export['Country ID'] = rid
            p_export["RegType"] = a 
           
            cagr = p_export
            cagr.columns = ['Year', 'lvalue', 'Country ID', 'RegType']
            cagr['value'] = math.exp(1) ** cagr['lvalue']                    
            cagr['test1'] = cagr['Year'] % 5
            def fun(x):
               if x == 2017:
                 return 0
               else:
                 return 1
            cagr['test2'] =cagr['Year'].apply(lambda x: fun(x))
            cagr['test'] = cagr['test1'] * cagr['test2']
            cagr = cagr.drop(cagr[cagr.test > 0].index)
            cagr['CAGR'] = (cagr['value'] / cagr['value'].shift(+1) ) ** (1/(cagr['Year'] - cagr['Year'].shift(+1))) - 1
            cagr = cagr.drop(['test1', 'test2', 'test'], axis = 1)
                    
            p_export = p_export.drop(['test1', 'test2', 'test'], axis = 1)

            df = pd.concat([df,entry2])
            df2 = pd.concat([df2,p_export])
            df3 = pd.concat([df3,cagr])
            
df.insert(1, "Sector", "ind")
df = df[df['Coe'].notna()]
ind2 = pd.merge(df,ind2country,on='Country ID',how='inner') 
ind2['CoeID']=ind2['CoeID'].str.replace('x0','xc')
ind2['CoeID']=ind2['CoeID'].str.replace('x1','xf')
ind2['CoeID']=ind2['CoeID'].str.replace('x2','xm')
ind2['CoeID']=ind2['CoeID'].str.replace('x3','xt')
ind2['CoeID']=ind2['CoeID'].str.replace('x4','xo')
ind2['tvalueID']=ind2['tvalueID'].str.replace('t_x0','t_xc')
ind2['tvalueID']=ind2['tvalueID'].str.replace('t_x1','t_xf')
ind2['tvalueID']=ind2['tvalueID'].str.replace('t_x2','t_xm')
ind2['tvalueID']=ind2['tvalueID'].str.replace('t_x3','t_xt')
ind2['tvalueID']=ind2['tvalueID'].str.replace('t_x4','t_xo')

df2["Sector"] = 'ind'
df2.columns = ['Year', 'lvalue', 'Country ID', 'RegType', 'Sector', 'value']        
             
df3["Sector"] = 'ind'
df3 = pd.merge(df3,indcountry,on='Country ID',how='inner') 

ind2_p = pd.merge(df2,ind2country,on='Country ID',how='inner') 

ind_regid = ind_regid.append({'RegType': 'lche + lfoo + lmac + ltex + loth', 'RegID': 'ind060'}, ignore_index=True)

ind2 = pd.merge(ind2,ind_regid,on='RegType',how='inner') 
ind2_p = pd.merge(ind2_p,ind_regid,on='RegType',how='inner') 
ind2_cagr = pd.merge(df3,ind_regid,on='RegType',how='inner') 


ind_regid["Sector"] = "ind"


endtime = datetime.datetime.now()
print (endtime - starttime)

print("Congradulations! Ind Finished :)")


# RES
data = table.drop(['lelc', 'lind', 'lcom', 'ltrn', 'lpet', 'lgdp_man', 'lche', 'lfoo', 'lmac', 'ltex', 'loth', 'lgasprice', 'loilprice'], axis = 1)

r_table = pd.read_excel(r'C:\Users\jiaxi\OneDrive\桌面\RBAC 2021\Demand Annual Builder\Input Scenarios.xlsx', sheet_name = 'RES')
rescountry = r_table.loc[:,['Country ID',  'Country']]

country_name = r_table['Country'].to_list()

ref_country = table['Country'].unique()

df = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()


for i in country_name:
    if i != 'ALL':
        if i in ref_country:

                Country = data[data['Country']== i]
                country_filter = r_table[r_table['Country']== i]
                
                retr_country = country_filter['Country'].to_list()[0]
                
                country_data = data[data['Country']==retr_country]

                x_labels = country_filter[['x0','x1','x2','x3','x4','x5','x6','x7', 'x8', 'x9',  'y']]
                rid = country_filter['Country ID'].tolist()[0]
                y_var = x_labels['y'].to_list()[0]

                def subcombs_2(dset):
                    data_ = []
                    for i in range(1,len(dset)+1):
                        for j in combinations(dset,i):
                            data_.append(list(j))
                    return data_


                def stitch(f):
                    out = ""
                    for k,i in enumerate(f):
                        if len(i) > 1:
                            out += i
                        else:
                            try:
                                out += i
                            except:
                                out +=i[0]
                        if k < len(f)-1:
                            out+=" + "
                    return out

                regressors = ['lgdp', 'lppl', 'lgdpp', 'luppl', 'llf']
                
                allcombs = subcombs_2(regressors)
                del allcombs[30:]
                allcombs.append(['lgdp', 'lppl', 'lgop'])
                allcombs.append(['lgdp', 'luppl', 'lgou'])
                allcombs.append(['lppl', 'luppl', 'lpou'])
                
                aclist = []

                for i in allcombs:
                     aclist.append('lres~'+stitch(i)) 

                for i in aclist:
                    model = smf.ols(i, data=country_data)
                    results = model.fit()
                    a = i.replace('lres~', '')
                    
                    values = pd.DataFrame(results.params).reset_index()
                    values.columns = ['names', 'coef']
                    xfinal = x_labels
                    xfinal['Intercept'] = 'Intercept'
                    xfinal = xfinal.T.reset_index()
                    xfinal.columns = ['labels', 'names']
                    export_val = xfinal.merge(values, on = 'names', how='left')
                    results.tvalues
                    t_stat = pd.DataFrame(results.tvalues).reset_index()
                    t_stat.columns = ['names', 't-statistic']
                    export_val1 = export_val.merge(t_stat, on = 'names', how='left')
                    
                    
                    r_columns = ['Year'] + values.xs("names", axis = 1).tolist()
                    del r_columns[1: 2]
                    forecast_var = country_data[r_columns].reset_index(drop=True)
                    forecast_var['Intercept'] = 1           
                    test = forecast_var[forecast_var['Year']>2016]
                    xval = test.drop(columns = ['Year'])

                    
                    entry2 = pd.DataFrame({'Country ID': [rid, rid, rid, rid,rid, rid, rid, rid, rid, rid, rid],
                                   'R-square': [results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared],
                                   'RegType': [a, a, a, a, a, a, a, a, a, a, a],
                                   'CoeID': ['intercept', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'],
                                   'Coe': [export_val1[export_val1['labels']=='Intercept']['coef'].to_list()[0], export_val1[export_val1['labels']=='x0']['coef'].to_list()[0], export_val1[export_val1['labels']=='x1']['coef'].to_list()[0], export_val1[export_val1['labels']=='x2']['coef'].to_list()[0], export_val1[export_val1['labels']=='x3']['coef'].to_list()[0], export_val1[export_val1['labels']=='x4']['coef'].to_list()[0], export_val1[export_val1['labels']=='x5']['coef'].to_list()[0], export_val1[export_val1['labels']=='x6']['coef'].to_list()[0], export_val1[export_val1['labels']=='x7']['coef'].to_list()[0], export_val1[export_val1['labels']=='x8']['coef'].to_list()[0], export_val1[export_val1['labels']=='x9']['coef'].to_list()[0]],
                                   'tvalueID': ['t_intercept', 't_x0', 't_x1', 't_x2', 't_x3', 't_x4', 't_x5', 't_x6', 't_x7', 't_x8', 't_x9'],
                                   'tvalue': [export_val1[export_val1['labels']=='Intercept']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x0']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x1']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x2']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x3']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x4']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x5']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x6']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x7']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x8']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x9']['t-statistic'].to_list()[0]]})
                    
                    y_hat = results.predict(xval) 
                    yr_p = yrs[yrs['Year']>=2017]            
                    p_export = pd.concat([yr_p, y_hat], axis=1, sort=False)
                    p_export['Country ID'] = rid
                    p_export["RegType"] = a
                    
                    cagr = p_export
                    cagr.columns = ['Year', 'lvalue', 'Country ID', 'RegType']
                    cagr['value'] = math.exp(1) ** cagr['lvalue']                    
                    cagr['test1'] = cagr['Year'] % 5
                    def fun(x):
                      if x == 2017:
                        return 0
                      else:
                        return 1
                    cagr['test2'] =cagr['Year'].apply(lambda x: fun(x))
                    cagr['test'] = cagr['test1'] * cagr['test2']
                    cagr = cagr.drop(cagr[cagr.test > 0].index)
                    cagr['CAGR'] = (cagr['value'] / cagr['value'].shift(+1) ) ** (1/(cagr['Year'] - cagr['Year'].shift(+1))) - 1
                    cagr = cagr.drop(['test1', 'test2', 'test'], axis = 1)
                    
                    p_export = p_export.drop(['test1', 'test2', 'test'], axis = 1)

                    df = pd.concat([df,entry2])
                    df2 = pd.concat([df2,p_export])
                    df3 = pd.concat([df3,cagr])
                    
            
df.insert(1, "Sector", "res")
df = df[df['Coe'].notna()]
df = pd.merge(df,rescountry,on='Country ID',how='inner') 

df2["Sector"] = 'res'
df2.columns = ['Year', 'lvalue', 'Country ID', 'RegType', 'Sector', 'value']             
df2 = pd.merge(df2,rescountry,on='Country ID',how='inner') 

df3["Sector"] = 'res'
df3 = pd.merge(df3,rescountry,on='Country ID',how='inner') 

allcombs2  = DataFrame(aclist)
allcombs2["ID"] =  range(len(allcombs2)) 
allcombs2["ID"] = allcombs2["ID"] + 1
allcombs2["Sector"] = "res"
allcombs2['RegID'] = allcombs2['ID'].apply(lambda x: 'res00' + str(x)  if x<10 else 'res0' +str(x) if x<100 else 'res' + str(x))
allcombs2.columns = ['RegType', 'ID', 'Sector', 'RegID']
allcombs2['RegType']  = allcombs2['RegType'].str.replace("lres~", "")
res_regid = allcombs2.drop(['ID','Sector'],axis = 1)

res = pd.merge(df,res_regid,on='RegType',how='inner') 
res_p = pd.merge(df2,res_regid,on='RegType',how='inner')
res_cagr = pd.merge(df3,res_regid,on='RegType',how='inner') 
res_regid["Sector"] = "res"


endtime = datetime.datetime.now()
print (endtime - starttime)

print("Congradulations! Res Finished :)")


# TRN
data = table.drop(['lelc', 'lind', 'lcom', 'lres', 'lpet','lgdp_man', 'lche', 'lfoo', 'lmac', 'ltex', 'loth', 'lgasprice', 'loilprice'], axis = 1)

r_table = pd.read_excel(r'C:\Users\jiaxi\OneDrive\桌面\RBAC 2021\Demand Annual Builder\Input Scenarios.xlsx', sheet_name = 'TRN')
trncountry = r_table.loc[:,['Country ID',  'Country']]

country_name = r_table['Country'].to_list()

ref_country = table['Country'].unique()

df = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()

for i in country_name:
    if i != 'ALL':
        if i in ref_country:

                Country = data[data['Country']== i]
                country_filter = r_table[r_table['Country']== i]
                
                retr_country = country_filter['Country'].to_list()[0]
                
                country_data = data[data['Country']==retr_country]

                x_labels = country_filter[['x0','x1','x2','x3','x4','x5','x6','x7', 'x8', 'x9',  'y']]
                rid = country_filter['Country ID'].tolist()[0]
                y_var = x_labels['y'].to_list()[0]

                def subcombs_2(dset):
                    data_ = []
                    for i in range(1,len(dset)+1):
                        for j in combinations(dset,i):
                            data_.append(list(j))
                    return data_


                def stitch(f):
                    out = ""
                    for k,i in enumerate(f):
                        if len(i) > 1:
                            out += i
                        else:
                            try:
                                out += i
                            except:
                                out +=i[0]
                        if k < len(f)-1:
                            out+=" + "
                    return out

                regressors = ['lgdp', 'lppl', 'lgdpp', 'luppl', 'llf']                
                allcombs = subcombs_2(regressors)
                del allcombs[30:]
                allcombs.append(['lgdp', 'lppl', 'lgop'])
                allcombs.append(['lgdp', 'luppl', 'lgou'])
                allcombs.append(['lppl', 'luppl', 'lpou'])
                
                aclist = []

                for i in allcombs:
                     aclist.append('ltrn~'+stitch(i))
                     

                for i in aclist:
                    model = smf.ols(i, data=country_data)
                    results = model.fit()
                    a = i.replace('ltrn~', '')
                    
                    values = pd.DataFrame(results.params).reset_index()
                    values.columns = ['names', 'coef']
                    xfinal = x_labels
                    xfinal['Intercept'] = 'Intercept'
                    xfinal = xfinal.T.reset_index()
                    xfinal.columns = ['labels', 'names']
                    export_val = xfinal.merge(values, on = 'names', how='left')
                    results.tvalues
                    t_stat = pd.DataFrame(results.tvalues).reset_index()
                    t_stat.columns = ['names', 't-statistic']
                    export_val1 = export_val.merge(t_stat, on = 'names', how='left')
                    
                    
                    r_columns = ['Year'] + values.xs("names", axis = 1).tolist()
                    del r_columns[1: 2]
                    forecast_var = country_data[r_columns].reset_index(drop=True)
                    forecast_var['Intercept'] = 1           
                    test = forecast_var[forecast_var['Year']>2016]
                    xval = test.drop(columns = ['Year'])

                    
                    entry2 = pd.DataFrame({'Country ID': [rid, rid, rid, rid,rid, rid, rid, rid, rid, rid, rid],
                                   'R-square': [results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared],
                                   'RegType': [a, a, a, a, a, a, a, a, a, a, a],
                                   'CoeID': ['intercept', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'],
                                   'Coe': [export_val1[export_val1['labels']=='Intercept']['coef'].to_list()[0], export_val1[export_val1['labels']=='x0']['coef'].to_list()[0], export_val1[export_val1['labels']=='x1']['coef'].to_list()[0], export_val1[export_val1['labels']=='x2']['coef'].to_list()[0], export_val1[export_val1['labels']=='x3']['coef'].to_list()[0], export_val1[export_val1['labels']=='x4']['coef'].to_list()[0], export_val1[export_val1['labels']=='x5']['coef'].to_list()[0], export_val1[export_val1['labels']=='x6']['coef'].to_list()[0], export_val1[export_val1['labels']=='x7']['coef'].to_list()[0], export_val1[export_val1['labels']=='x8']['coef'].to_list()[0], export_val1[export_val1['labels']=='x9']['coef'].to_list()[0]],
                                   'tvalueID': ['t_intercept', 't_x0', 't_x1', 't_x2', 't_x3', 't_x4', 't_x5', 't_x6', 't_x7', 't_x8', 't_x9'],
                                   'tvalue': [export_val1[export_val1['labels']=='Intercept']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x0']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x1']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x2']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x3']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x4']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x5']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x6']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x7']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x8']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x9']['t-statistic'].to_list()[0]]})
                    
                    y_hat = results.predict(xval) 
                    yr_p = yrs[yrs['Year']>=2017]            
                    p_export = pd.concat([yr_p, y_hat], axis=1, sort=False)
                    p_export['Country ID'] = rid
                    p_export["RegType"] = a
                    
                    cagr = p_export
                    cagr.columns = ['Year', 'lvalue', 'Country ID', 'RegType']
                    cagr['value'] = math.exp(1) ** cagr['lvalue']                    
                    cagr['test1'] = cagr['Year'] % 5
                    def fun(x):
                      if x == 2017:
                        return 0
                      else:
                        return 1
                    cagr['test2'] =cagr['Year'].apply(lambda x: fun(x))
                    cagr['test'] = cagr['test1'] * cagr['test2']
                    cagr = cagr.drop(cagr[cagr.test > 0].index)
                    cagr['CAGR'] = (cagr['value'] / cagr['value'].shift(+1) ) ** (1/(cagr['Year'] - cagr['Year'].shift(+1))) - 1
                    cagr = cagr.drop(['test1', 'test2', 'test'], axis = 1)
                    
                    p_export = p_export.drop(['test1', 'test2', 'test'], axis = 1)

                    df = pd.concat([df,entry2])
                    df2 = pd.concat([df2,p_export])
                    df3 = pd.concat([df3,cagr])
                    
            
df.insert(1, "Sector", "trn")
df = df[df['Coe'].notna()]
df = pd.merge(df,trncountry,on='Country ID',how='inner') 

df2["Sector"] = 'trn'
df2.columns = ['Year', 'lvalue', 'Country ID', 'RegType', 'Sector', 'value']             
df2 = pd.merge(df2,trncountry,on='Country ID',how='inner') 

df3["Sector"] = 'trn'
df3 = pd.merge(df3,trncountry,on='Country ID',how='inner') 

allcombs2  = DataFrame(aclist)
allcombs2["ID"] =  range(len(allcombs2)) 
allcombs2["ID"] = allcombs2["ID"] + 1
allcombs2["Sector"] = "trn"
allcombs2['RegID'] = allcombs2['ID'].apply(lambda x: 'trn00' + str(x)  if x<10 else 'trn0' +str(x) if x<100 else 'trn' + str(x))
allcombs2.columns = ['RegType', 'ID', 'Sector', 'RegID']
allcombs2['RegType']  = allcombs2['RegType'].str.replace("ltrn~", "")
trn_regid = allcombs2.drop(['ID','Sector'],axis = 1)

trn = pd.merge(df,trn_regid,on='RegType',how='inner') 
trn_p = pd.merge(df2,trn_regid,on='RegType',how='inner')
trn_cagr = pd.merge(df3,trn_regid,on='RegType',how='inner') 
trn_regid["Sector"] = "trn"

endtime = datetime.datetime.now()
print (endtime - starttime)

print("Congradulations! Trn Finished :)")



# PET
data = table.drop(['lelc', 'lind', 'lcom', 'lres', 'ltrn','lgdp_man',  'lfoo', 'lmac', 'ltex', 'loth', 'lgop', 'lgou', 'lpou'], axis = 1)

r_table = pd.read_excel(r'C:\Users\jiaxi\OneDrive\桌面\RBAC 2021\Demand Annual Builder\Input Scenarios.xlsx', sheet_name = 'PET')
petcountry = r_table.loc[:,['Country ID',  'Country']]

country_name = r_table['Country'].to_list()

ref_country = table['Country'].unique()

df = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()

for i in country_name:
    if i != 'ALL':
        if i in ref_country:

                Country = data[data['Country']== i]
                country_filter = r_table[r_table['Country']== i]
                
                retr_country = country_filter['Country'].to_list()[0]
                
                country_data = data[data['Country']==retr_country]

                x_labels = country_filter[['x0','x1','xc','xpg','xpo','x5','x6','x7', 'x8', 'x9',  'y']]
                rid = country_filter['Country ID'].tolist()[0]
                y_var = x_labels['y'].to_list()[0]

                def subcombs_2(dset):
                    data_ = []
                    for i in range(1,len(dset)+1):
                        for j in combinations(dset,i):
                            data_.append(list(j))
                    return data_


                def stitch(f):
                    out = ""
                    for k,i in enumerate(f):
                        if len(i) > 1:
                            out += i
                        else:
                            try:
                                out += i
                            except:
                                out +=i[0]
                        if k < len(f)-1:
                            out+=" + "
                    return out

                regressors = ['lgdp', 'lppl', 'lche', 'lgasprice', 'loilprice']
                allcombs = subcombs_2(regressors)               
                del allcombs[55:]
                
                aclist = []

                for i in allcombs:
                     aclist.append('lpet~'+stitch(i)) 

                for i in aclist:
                    model = smf.ols(i, data=country_data)
                    results = model.fit()
                    a = i.replace('lpet~', '')
                    
                    values = pd.DataFrame(results.params).reset_index()
                    values.columns = ['names', 'coef']
                    xfinal = x_labels
                    xfinal['Intercept'] = 'Intercept'
                    xfinal = xfinal.T.reset_index()
                    xfinal.columns = ['labels', 'names']
                    export_val = xfinal.merge(values, on = 'names', how='left')
                    results.tvalues
                    t_stat = pd.DataFrame(results.tvalues).reset_index()
                    t_stat.columns = ['names', 't-statistic']
                    export_val1 = export_val.merge(t_stat, on = 'names', how='left')
                    
                    
                    r_columns = ['Year'] + values.xs("names", axis = 1).tolist()
                    del r_columns[1: 2]
                    forecast_var = country_data[r_columns].reset_index(drop=True)
                    forecast_var['Intercept'] = 1           
                    test = forecast_var[forecast_var['Year']>2016]
                    xval = test.drop(columns = ['Year'])

                    
                    entry2 = pd.DataFrame({'Country ID': [rid, rid, rid, rid,rid, rid, rid, rid, rid, rid, rid],
                                   'R-square': [results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared, results.rsquared],
                                   'RegType': [a, a, a, a, a, a, a, a, a, a, a],
                                   'CoeID': ['intercept', 'x0', 'x1', 'xc', 'xpg', 'xpo', 'x5', 'x6', 'x7', 'x8', 'x9'],
                                   'Coe': [export_val1[export_val1['labels']=='Intercept']['coef'].to_list()[0], export_val1[export_val1['labels']=='x0']['coef'].to_list()[0], export_val1[export_val1['labels']=='x1']['coef'].to_list()[0], export_val1[export_val1['labels']=='xc']['coef'].to_list()[0], export_val1[export_val1['labels']=='xpg']['coef'].to_list()[0], export_val1[export_val1['labels']=='xpo']['coef'].to_list()[0], export_val1[export_val1['labels']=='x5']['coef'].to_list()[0], export_val1[export_val1['labels']=='x6']['coef'].to_list()[0], export_val1[export_val1['labels']=='x7']['coef'].to_list()[0], export_val1[export_val1['labels']=='x8']['coef'].to_list()[0], export_val1[export_val1['labels']=='x9']['coef'].to_list()[0]],
                                   'tvalueID': ['t_intercept', 't_x0', 't_x1', 't_xc', 't_xpg', 't_xpo', 't_x5', 't_x6', 't_x7', 't_x8', 't_x9'],
                                   'tvalue': [export_val1[export_val1['labels']=='Intercept']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x0']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x1']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='xc']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='xpg']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='xpo']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x5']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x6']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x7']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x8']['t-statistic'].to_list()[0], export_val1[export_val1['labels']=='x9']['t-statistic'].to_list()[0]]})
                    
                    y_hat = results.predict(xval) 
                    yr_p = yrs[yrs['Year']>=2017]            
                    p_export = pd.concat([yr_p, y_hat], axis=1, sort=False)
                    p_export['Country ID'] = rid
                    p_export["RegType"] = a
                    
                    cagr = p_export
                    cagr.columns = ['Year', 'lvalue', 'Country ID', 'RegType']
                    cagr['value'] = math.exp(1) ** cagr['lvalue']                    
                    cagr['test1'] = cagr['Year'] % 5
                    def fun(x):
                      if x == 2017:
                        return 0
                      else:
                        return 1
                    cagr['test2'] =cagr['Year'].apply(lambda x: fun(x))
                    cagr['test'] = cagr['test1'] * cagr['test2']
                    cagr = cagr.drop(cagr[cagr.test > 0].index)
                    cagr['CAGR'] = (cagr['value'] / cagr['value'].shift(+1) ) ** (1/(cagr['Year'] - cagr['Year'].shift(+1))) - 1
                    cagr = cagr.drop(['test1', 'test2', 'test'], axis = 1)
                    
                    p_export = p_export.drop(['test1', 'test2', 'test'], axis = 1)

                    df = pd.concat([df,entry2])
                    df2 = pd.concat([df2,p_export])
                    df3 = pd.concat([df3,cagr])
                                  
                    
                    
            
df.insert(1, "Sector", "pet")
df = df[df['Coe'].notna()]
df = pd.merge(df,petcountry,on='Country ID',how='inner') 

df2["Sector"] = 'pet'
df2.columns = ['Year', 'lvalue', 'Country ID', 'RegType', 'Sector', 'value']             
df2 = pd.merge(df2,petcountry,on='Country ID',how='inner') 

df3["Sector"] = 'pet'
df3 = pd.merge(df3,petcountry,on='Country ID',how='inner') 


allcombs2  = DataFrame(aclist)
allcombs2["ID"] =  range(len(allcombs2)) 
allcombs2["ID"] = allcombs2["ID"] + 1
allcombs2["Sector"] = "pet"
allcombs2['RegID'] = allcombs2['ID'].apply(lambda x: 'pet00' + str(x)  if x<10 else 'pet0' +str(x) if x<100 else 'pet' + str(x))
allcombs2.columns = ['RegType', 'ID', 'Sector', 'RegID']
allcombs2['RegType']  = allcombs2['RegType'].str.replace("lpet~", "")
pet_regid = allcombs2.drop(['ID','Sector'],axis = 1)

pet = pd.merge(df,pet_regid,on='RegType',how='inner') 
pet_p = pd.merge(df2,pet_regid,on='RegType',how='inner') 
pet_cagr = pd.merge(df3,pet_regid,on='RegType',how='inner') 
pet_regid["Sector"] = "pet"


endtime = datetime.datetime.now()
print (endtime - starttime)

print("Congradulations! Pet Finished :)")

#############

total = pd.DataFrame()
total = [com, ind, ind2, res, trn, pet]
total2 = pd.concat(total,ignore_index=True)
order = ['Country ID', 'Country', 'R-square', 'RegType', 'RegID', 'CoeID', 'Coe', 'tvalueID', 'tvalue']
total2 =  total2.reindex(columns = order)

total_forecast_p1 = pd.DataFrame()
total_forecast_p1 = [com_p, ind_p, ind2_p, res_p, trn_p, pet_p]
total_forecast2_p1 = pd.concat(total_forecast_p1,ignore_index=True)
order = ['Country ID','Country', 'Sector', 'RegType', 'RegID', 'Year', 'lvalue']
total_forecast2_p1 = total_forecast2_p1.reindex(columns = order)

total_cagr = pd.DataFrame()
total_cagr = [com_cagr, ind_cagr, ind2_cagr, res_cagr, trn_cagr, pet_cagr]
total_cagr = pd.concat(total_cagr,ignore_index=True)
order = ['Country ID','Country', 'Sector', 'RegType', 'RegID', 'Year', 'lvalue', 'value', 'CAGR']
total_cagr = total_cagr.reindex(columns = order)


'''
total_forecast_p2 = pd.DataFrame()
total_forecast_p2 = [ind_p, ind2_p]
total_forecast2_p2 = pd.concat(total_forecast_p2,ignore_index=True)
order = ['Country ID','Country', 'Sector', 'RegType', 'RegID', 'Year', 'lvalue']
total_forecast2_p2 = total_forecast2_p2.reindex(columns = order)

total_forecast_p3 = pd.DataFrame()
total_forecast_p3 = [res_p, trn_p]
total_forecast2_p3 = pd.concat(total_forecast_p3,ignore_index=True)
order = ['Country ID','Country', 'Sector', 'RegType', 'RegID', 'Year', 'lvalue']
total_forecast2_p3 = total_forecast2_p3.reindex(columns = order)
'''
RegID = pd.DataFrame()
RegID = [com_regid, ind_regid, res_regid, trn_regid, pet_regid]
RegID2 = pd.concat(RegID,ignore_index=True)

total2.to_csv(r'C:\Users\jiaxi\OneDrive\桌面\RBAC 2021\Demand Annual Builder\Total Results 2021 0702.csv',index=None)
total_forecast2_p1.to_csv(r'C:\Users\jiaxi\OneDrive\桌面\RBAC 2021\Demand Annual Builder\Total Forecast 2021 0702.csv',index=None)
# total_forecast2_p2.to_csv(r'C:\Users\jiaxi\OneDrive\桌面\RBAC 2021\Demand Annual Builder\Total Forecast Part 2 2021 0702.csv',index=None)
# total_forecast2_p3.to_csv(r'C:\Users\jiaxi\OneDrive\桌面\RBAC 2021\Demand Annual Builder\Total Forecast Part 3 2021 0702.csv',index=None)
RegID2.to_csv(r'C:\Users\jiaxi\OneDrive\桌面\RBAC 2021\Demand Annual Builder\Total RegID 2021 0702.csv',index=None)
total_cagr.to_csv(r'C:\Users\jiaxi\OneDrive\桌面\RBAC 2021\Demand Annual Builder\Total CAGR 2021 0702.csv',index=None)


endtime = datetime.datetime.now()
print (endtime - starttime)

print("Congradulations! Finished :)")





