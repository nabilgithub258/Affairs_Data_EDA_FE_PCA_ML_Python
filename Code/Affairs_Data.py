#!/usr/bin/env python
# coding: utf-8

# In[1]:


############################################################################################################################
#### We are currently working on a dataset related to female affairs, driven purely by our passion for statistics and ######
#### data analysis. We want to emphasize that there is no bias in our approach, and our work is not intended to invite #####
#### assumptions or judgements about the subject matter. Our goal is to explore the data objectively and rigorously. #######
############################################################################################################################


# In[2]:


#####################################################################################################
######################### FEMALE AFFAIRS DATA SET  ##################################################
#####################################################################################################


# In[3]:


#################################################################
############ Part I - Importing
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('Affairs.csv')                      #### getting the data


# In[5]:


df.head(10)


# In[6]:


df.info()


# In[7]:


#####################################################################
########################### Part II - Duplicates
#####################################################################


# In[8]:


df[df.duplicated()]                  #### no duplicates


# In[9]:


####################################################################
############## Part III - Missing Values
####################################################################


# In[10]:


from matplotlib.colors import LinearSegmentedColormap

Amelia = LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])


# In[11]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')

#### why Amelia, if you coming from R then you might have used Amelia package which detects the missing value 
#### On July 2, 1937, Amelia disappeared over the Pacific Ocean while attempting to become the first female pilot to circumnavigate the world


# In[12]:


df.isnull().any()                         #### no null data


# In[13]:


####################################################################
############## Part IV - Feature Engineering
####################################################################


# In[14]:


'''
rate_marriage   : How rate marriage, 1 = very poor, 2 = poor, 3 = fair,
                4 = good, 5 = very good
age             : Age
yrs_married     : No. years married. Interval approximations. See
                original paper for detailed explanation.
children        : No. children
religious       : How relgious, 1 = not, 2 = mildly, 3 = fairly,
                4 = strongly
educ            : Level of education, 9 = grade school, 12 = high
                school, 14 = some college, 16 = college graduate,
                17 = some graduate school, 20 = advanced degree
occupation      : 1 = student, 2 = farming, agriculture; semi-skilled,
                or unskilled worker; 3 = white-colloar; 4 = teacher
                counselor social worker, nurse; artist, writers;
                technician, skilled worker, 5 = managerial,
                administrative, business, 6 = professional with
                advanced degree
occupation_husb : Husband's occupation. Same as occupation.
affairs         : measure of time spent in extramarital affairs

'''


# In[15]:


df.head()


# In[16]:


df.columns


# In[17]:


df.drop(columns='Unnamed: 0',inplace=True)               #### for obvious reasons


# In[18]:


df.head()              #### now we will take on rate_marriage, 0 as poor,
                       #### 1 as good
                       #### 2 excellent


# In[19]:


df.rate_marriage.value_counts()                  #### so here 1 and 2 as 0
                                                 #### 3 as 1
                                                 #### 4 and 5 as 2


# In[20]:


df.rate_marriage = df.rate_marriage.apply(lambda x: 0 if x in [1, 2] else (1 if x == 3 else 2))


# In[21]:


df.head()                        #### now it makes more sense


# In[22]:


df.rate_marriage.value_counts()                  #### note: 0 - poor, 1 - good, 2 - excellent


# In[23]:


df.yrs_married.value_counts()                   #### this doesn't make sense


# In[24]:


df.yrs_married = df.yrs_married.round()


# In[25]:


df.yrs_married.value_counts()                       #### we will make it easier to read
                                                    #### 0 as 0-5 years married
                                                    #### 1 as 5-10 years married
                                                    #### 2 as above 10 years married


# In[26]:


df['years_married'] = df['yrs_married'].map({2:'0-5 years',
                                             6:'5-10 years',
                                             16:'above 10 years',
                                             23:'above 10 years',
                                             9:'5-10 years',
                                             13:'above 10 years',
                                             0:'0-5 years'})


# In[27]:


df.head()


# In[28]:


df.years_married.value_counts()


# In[29]:


df.children.value_counts()


# In[30]:


df.children = df.children.round()


# In[31]:


df.children.value_counts()                       #### we will do into 3 cats, 0 kids = 0
                                                 #### 1-2 kids = 1
                                                 #### more then 2 kids = 2


# In[32]:


df['kids'] = df.children.map({0:'zero',
                              2:'1-2 kids',
                              1:'1-2 kids',
                              3:'more then 2',
                              4:'more then 2',
                              6:'more then 2'})


# In[33]:


df.head()


# In[34]:


df.kids.value_counts()


# In[35]:


df.religious.value_counts()                     #### with this we will do, 1 - not religious
                                                #### 2-3 neutral
                                                #### 4 as religious


# In[36]:


df['religion'] = df.religious.map({3:'Neutral',
                                   2:'Neutral',
                                   1:'Not Religious',
                                   4:'Religious'})


# In[37]:


df.head()


# In[38]:


df.educ.value_counts()


# In[39]:


df['education'] = df.educ.map({14:'School',
                               12:'School',
                               16:'College',
                               17:'College',
                               20:'University',
                                9:'School'})


# In[40]:


df.head()


# In[41]:


df.education.value_counts()


# In[42]:


df.occupation.value_counts()


# In[43]:


df['occupations'] = df.occupation.map({3:'white collar',
                                       4:'skilled worker',
                                       2:'unskilled worker',
                                       5:'professional job',
                                       6:'advanced job',
                                       1:'student'})


# In[44]:


df.head()


# In[45]:


df.occupations.value_counts()


# In[46]:


df['husb_occupation'] = df.occupation_husb.map({3:'white collar',
                                       4:'skilled worker',
                                       2:'unskilled worker',
                                       5:'professional job',
                                       6:'advanced job',
                                       1:'student'})


# In[47]:


df.head()


# In[48]:



df['affair_binary'] = df['affairs'].round()


# In[49]:


df.head()


# In[50]:


df.affair_binary.value_counts()


# In[51]:


df['target'] = df.affair_binary.apply(lambda x: 0 if x==0 else 1)


# In[52]:


df.target.value_counts()


# In[53]:


######################################################################
############## Part V - EDA
######################################################################


# In[54]:


df.groupby('target').mean()

#### from this we can derive that people who had affairs were less happy in marriage, older, more years married, more kids, less religious, less educated compared to people who didnt have affairs


# In[55]:


df['affairs'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Affairs Age Graph')

plt.xlabel('Number of people')

plt.ylabel('Density')

#### seems like most of the people didn't have affairs


# In[56]:


df.head()


# In[57]:


df.age.mean()


# In[58]:


df.age.std()                  #### seems good


# In[59]:


corr = df.corr()

corr


# In[60]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(corr,ax=ax,linewidths=0.5,annot=True,cmap='viridis')

#### we will took more deeper into occupation and more correlated cols


# In[61]:


custom = {0:'black',
         1:'red'}

g = sns.jointplot(x=df.occupation,y=df.age,data=df,hue='target',palette=custom)

g.fig.set_size_inches(17,9)

#### seems like theres a peak in affairs in occupation 3 and age between 25-30


# In[62]:


df.head()


# In[63]:


sns.catplot(x='target',y='yrs_married',data=df,kind='box',height=7,aspect=2,legend=True,hue='kids',palette='Set2')

#### this is quite revealing, people had affairs with zero kids while compared to who didn't have affairs, apart from this its pretty even on other cats


# In[64]:


custom = {'Neutral':'black',
          'Not Religious':'blue',
          'Religious':'red'}

g = sns.jointplot(x='target',y='rate_marriage',data=df,hue='religion',kind='kde',fill=True,palette=custom)

g.fig.set_size_inches(17,9)

#### people who rated 0 for marriage and had affairs were not religious
#### but if you really see and absorb then you will realize people who had affairs were more on the neutral side of religion


# In[65]:


sns.catplot(x='target',y='yrs_married',data=df,kind='box',height=7,aspect=2,legend=True,hue='education',palette='Set2')

#### quite interesting that people who had affairs were majority from University level education but on the higher side on the years married scale


# In[66]:


g = sns.jointplot(x='age',y='target',data=df,kind='reg',x_bins=[range(1,80)],color='black',joint_kws={'line_kws':{'color':'red'}})

g.fig.set_size_inches(17,9)

#### clearly we see the linear correlation between them, more older meaning more likely to cheat from the data we have


# In[67]:


from scipy.stats import pearsonr


# In[68]:


co_eff, p_value = pearsonr(df.age,df.target)

co_eff


# In[69]:


p_value                          #### definately related


# In[70]:


custom = {'5-10 years':'purple',
          'above 10 years':'pink',
          '0-5 years':'green'}

sns.catplot(x='occupations',y='affairs',data=df,kind='strip',height=7,aspect=2,legend=True,hue='years_married',jitter=True,palette=custom)

#### this gives us more better picture to people who had affairs and their occupations and years married, pretty revealing


# In[71]:


df.head()


# In[72]:


custom = {0:'purple',
          1:'green'}

sns.catplot(x='education',data=df,hue='target',kind='count',palette=custom,height=7,aspect=2)

#### seems like people who were in school had more affairs on the count level


# In[73]:


pl = sns.FacetGrid(df,hue='education',aspect=4,height=4)

pl.map(sns.kdeplot,'age',fill=True)

pl.set(xlim=(15,df.age.max()))

pl.add_legend()

#### college has the most density in this data set with the range of 25-30 years old


# In[74]:


custom = {0:'black',
          1:'green'}

pl = sns.FacetGrid(df,hue='target',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'age',fill=True)

pl.set(xlim=(15,df.age.max()))

pl.add_legend()

#### people who are young in their year range of 20-25 dont tend to have affairs compared to other age ranges


# In[75]:




pl = sns.catplot(y='target',x='education',data=df,kind='point',hue='kids',height=10,aspect=2,palette='Dark2')

#### the biggest leap we see is the university educated and having more then 2 kids


# In[76]:


sns.lmplot(x='age',y='target',data=df,hue='kids',palette='Set2',x_bins=[range(1,80)],height=6,aspect=2)

#### the very clear linear relationship I see is zero kids and age


# In[77]:




sns.lmplot(x='age',y='target',data=df,hue='occupations',x_bins=[range(1,80)],height=7,aspect=2)

#### definately we do see some linear relationship here


# In[78]:


custom = {0:'black',
          1:'green'}

sns.lmplot(x='age',y='children',data=df,hue='target',x_bins=[range(15,80)],height=7,aspect=2,palette=custom)

#### really interesting, both targets have the same or similar linear relationship to years married and age


# In[79]:


df.head()


# In[80]:


df.groupby(['education','kids'])['target'].sum().unstack().unstack().plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',markersize=15,linestyle='dashed',linewidth=4,color='red')

#### the most affairs are from people who are school educated with 1-2 kids


# In[81]:


df.groupby(['education','kids'])['target'].sum().unstack()


# In[82]:


heat = df.groupby(['years_married','kids','religion','education','occupations'])['target'].sum().unstack().unstack().unstack().fillna(0)

heat


# In[83]:


fig, ax = plt.subplots(figsize=(30,25)) 

sns.heatmap(heat,linewidths=0.1,ax=ax,cmap='viridis')

#### this is quite interesting, we can derive a lot of information from this


# In[84]:


#### We have done enough for EDA, now lets move to modelling part


# In[85]:


######################################################################
############## Part VI - Classification
######################################################################


# In[86]:


df.head()


# In[87]:


X = df.drop(columns=['affairs','years_married','kids','religion','education','occupations','husb_occupation','affair_binary'])

X.head()

#### first we will do without doing any much changes to the data set


# In[88]:


corr = X.corr()


# In[89]:


fig, ax = plt.subplots(figsize=(20,7)) 

sns.heatmap(corr,annot=True,linewidths=0.5,ax=ax,cmap='cividis')


# In[90]:


from sklearn.model_selection import train_test_split


# In[91]:


X.drop(columns='target',inplace=True)

X.head()


# In[92]:


y = df.target

y.head()


# In[93]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[94]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# In[95]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['rate_marriage', 'children', 'religious', 'educ',
                                   'occupation', 'occupation_husb','age','yrs_married'])
    ])


# In[96]:


from sklearn.pipeline import Pipeline


# In[97]:


from sklearn.linear_model import LogisticRegression


# In[98]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


# In[99]:


model.fit(X_train,y_train)


# In[100]:


y_predict = model.predict(X_test)


# In[101]:


from sklearn import metrics


# In[102]:


metrics.accuracy_score(y_test,y_predict)


# In[103]:


print(metrics.classification_report(y_test,y_predict))              #### not a bad model, our data size is too small and is very unbalanced with the ratio hence we see this


# In[104]:


cm = metrics.confusion_matrix(y_test,y_predict)

labels = ['No Affairs','Affairs']

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)

fig, ax = plt.subplots(figsize=(25,12))

disp.plot(ax=ax)


# In[105]:


#### now we will do by taking care of any multicollinearity

from statsmodels.tools.tools import add_constant

X_with_constant = add_constant(X)

X_with_constant.head()                    #### setting up Vif


# In[106]:


vif = pd.DataFrame()                      #### this is extremely helpful and important to know which col can be a problem


# In[107]:


vif["Feature"] = X_with_constant.columns


# In[108]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif["VIF"] = [variance_inflation_factor(X_with_constant.values, i) for i in range(X_with_constant.shape[1])]


# In[109]:


vif                         #### rule of thumb is to drop any cols which are above 5 as they can lead to multicollinearity, but I prefer the other way around


# In[110]:


from sklearn.decomposition import PCA                 #### love PCA


# In[111]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components='mle'))  
        ]), ['rate_marriage', 'age', 'yrs_married', 'children', 'religious', 'educ', 'occupation', 'occupation_husb'])
    ])


# In[112]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


# In[113]:


model.fit(X_train,y_train)


# In[114]:


y_predict = model.predict(X_test)


# In[115]:


print(metrics.classification_report(y_test,y_predict))


# In[116]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['rate_marriage', 'children', 'religious', 'educ',
                                   'occupation', 'occupation_husb','age','yrs_married'])
    ])


# In[117]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


# In[118]:


model.fit(X_train,y_train)


# In[119]:


y_predict = model.predict(X_test)


# In[120]:


print(metrics.classification_report(y_test,y_predict))


# In[121]:


from sklearn.linear_model import RidgeClassifier           #### Now we will see the same with Ridge


# In[122]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['rate_marriage', 'children', 'religious', 'educ',
                                   'occupation', 'occupation_husb','age','yrs_married'])
    ])


# In[123]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RidgeClassifier(alpha=1.0))
])


# In[124]:


model.fit(X_train,y_train)


# In[125]:


y_predict = model.predict(X_test)


# In[126]:


metrics.accuracy_score(y_test,y_predict)


# In[127]:


print(metrics.classification_report(y_test,y_predict))            #### even with ridge classifier we are getting ok result
                                                                  #### the problem is the support and how small our data set is to train the model


# In[128]:


df.head()                        #### now we will introduce the feature engineered cols and drop the corresponding numerical cols


# In[129]:


X = df.drop(columns=['yrs_married','children','religious','educ','occupation','occupation_husb','affairs','affair_binary','target'])


# In[130]:


X.head()


# In[131]:


y = df['target']

y.head()


# In[132]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[133]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components='mle'))  
        ]), ['rate_marriage', 'age']),
        ('cat', OneHotEncoder(drop='first'), ['years_married', 'kids','religion','education','occupations','husb_occupation'])

    ])


# In[134]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


# In[135]:


model.fit(X_train,y_train)


# In[136]:


y_predict = model.predict(X_test)


# In[137]:


print(metrics.classification_report(y_test,y_predict))            #### not much difference between this version to the original version we had


# In[138]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['rate_marriage', 'age']),
        ('cat', OneHotEncoder(drop='first'), ['years_married', 'kids','religion','education','occupations','husb_occupation'])

    ])

#### lets see without PCA


# In[139]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


# In[140]:


model.fit(X_train,y_train)


# In[141]:


y_predict = model.predict(X_test)


# In[142]:


print(metrics.classification_report(y_test,y_predict))


# In[143]:


from sklearn.ensemble import RandomForestClassifier

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])


# In[144]:


model.fit(X_train,y_train)


# In[145]:


y_predict = model.predict(X_test)


# In[146]:


print(metrics.classification_report(y_test,y_predict))               #### no wonder random forest is not helping here because our data set is so imbalanced


# In[147]:


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['years_married', 'kids','religion','education','occupations','husb_occupation']),
        ('num', StandardScaler(), ['rate_marriage','age'])
    ])


# In[148]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RidgeClassifier(alpha=1.0))
])


# In[149]:


model.fit(X_train,y_train)


# In[150]:


y_predict = model.predict(X_test)


# In[151]:


metrics.accuracy_score(y_test,y_predict)


# In[152]:


print(metrics.classification_report(y_test,y_predict))      


# In[153]:


from sklearn.model_selection import GridSearchCV


# In[154]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])


# In[155]:


param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

model_grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',verbose=3)
model_grid.fit(X_train, y_train)


# In[156]:


best_model = model_grid.best_estimator_


# In[157]:


y_predict = best_model.predict(X_test)


# In[158]:


print(metrics.classification_report(y_test,y_predict))               #### the model is not improving beyond this point it seems so we will halt further phase at this point


# In[159]:


from imblearn.over_sampling import ADASYN        #### one last try


# In[160]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


# In[161]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components='mle'))  
        ]), ['rate_marriage', 'age']),
        ('cat', OneHotEncoder(drop='first'), ['years_married', 'kids','religion','education','occupations','husb_occupation'])

    ])


# In[162]:


from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier


# In[163]:


model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('resample', ADASYN(random_state=42)),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])


# In[164]:


param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7],
    'classifier__colsample_bytree': [0.3, 0.7],
    'classifier__subsample': [0.5, 0.8]
}


# In[165]:


from sklearn.model_selection import RandomizedSearchCV


# In[166]:


get_ipython().run_cell_magic('time', '', "\nrandom_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=50, scoring='accuracy', cv=5, verbose=2, random_state=42)\nrandom_search.fit(X_train, y_train)")


# In[167]:


best_model = random_search.best_estimator_


# In[168]:


y_predict = best_model.predict(X_test)


# In[169]:


print(metrics.classification_report(y_test,y_predict))            #### made it worst honestly


# In[170]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_jobs=-1))
])


# In[171]:


param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

model_grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',verbose=3)
model_grid.fit(X_train, y_train)


# In[172]:


best_model = model_grid.best_estimator_


# In[173]:


y_predict = best_model.predict(X_test)


# In[174]:


print(metrics.classification_report(y_test,y_predict))           #### we will move to another affair dataset which contains both female and male


# In[175]:


######################################################################
############## Part VII - Affairs_2
######################################################################


# In[176]:


df = pd.read_csv('Affairs_2.csv')                  #### we will not be doing much EDA on this one, straight to the point


# In[177]:


df.head()


# In[178]:


df.drop(columns='Unnamed: 0',inplace=True)

df.head()


# In[179]:


corr = df.corr()


# In[180]:


fig, ax = plt.subplots(figsize=(20,7)) 

sns.heatmap(corr,annot=True,linewidths=0.5,ax=ax,cmap='viridis')


# In[181]:


df.isnull().any()


# In[182]:


df[df.duplicated()]


# In[183]:


df = df.drop_duplicates()


# In[184]:


df[df.duplicated()]


# In[185]:


df.info()


# In[186]:


df.affairs.value_counts()


# In[187]:


df['Affairs'] = df.affairs.apply(lambda x:0 if x==0 else 1)


# In[188]:


df.head()


# In[189]:


df.Affairs.value_counts()                  #### this will be a problem, low data set and imbalanced target values


# In[190]:


custom = {'male':'black',
          'female':'pink'}

g = sns.jointplot(x=df['age'],y=df['yearsmarried'],data=df,hue='gender',palette=custom)

g.fig.set_size_inches(17,9)

#### clearly we can see that females are slightly more on the years married side


# In[191]:


custom = {'male':'black',
          'female':'pink'}

sns.lmplot(x='yearsmarried',y='Affairs',data=df,hue='gender',height=7,aspect=2,palette=custom,x_bins=[range(0,15)])

#### seems like both genders have correlation to affairs and years they been married while males are marginally more likely then females


# In[192]:


df.head()


# In[193]:


df.age.value_counts()


# In[194]:


df.yearsmarried.value_counts()


# In[195]:


df['years_married'] = df.yearsmarried.apply(lambda x:1 if x<1 else x)


# In[196]:


df.head()


# In[197]:


df.years_married.value_counts()


# In[198]:


df.religiousness.value_counts()


# In[199]:


df['religion'] = df.religiousness.apply(lambda x:0 if x in [1,2] else (1 if x==3 else 2))


# In[200]:


df.religion.value_counts()             #### 0 not religious, 1 neutral and 2 is religious


# In[201]:


##### Level of education. Coding: 9 = grade school, 12 = high school graduate, 
#### 14 = some college, 16 = college graduate, 
#### 17 = some graduate work, 18 = master's degree, 
#### 20 = Ph.D., M.D., or other advanced degree.


# In[202]:


df['Education'] = df.education.apply(lambda x: 0 if x in [9,12] else (1 if x in [14,16] else (2 if x in [17,18] else 3)))


# In[203]:


df.Education.value_counts()


# In[204]:


df.head()


# In[205]:


df.occupation.value_counts()


# In[206]:


df.rating.value_counts()


# In[207]:


df['Rating'] = df.rating.apply(lambda x:0 if x in [1,2] else (1 if x==3 else 2))


# In[208]:


df.Rating.value_counts()               #### 0 not happy, 1 neutral, 2 not happy


# In[209]:


custom = {0:'red',
          1:'black',
          2:'green'}

sns.lmplot(x='yearsmarried',y='Affairs',data=df,hue='Rating',height=7,aspect=2,x_bins=[range(0,15)],palette=custom)

#### doesnt suprise us at all, if you not happy in marriage you tend to have affairs according to this very small data set we have, definately we not making any claims


# In[210]:


df.head()


# In[545]:


X = df.drop(columns=['affairs','yearsmarried','religiousness','education','occupation','rating'])

X.head()


# In[546]:


X = X[['Rating','age','gender','children','years_married','religion','Education']]

X.head()


# In[548]:


y = df.Affairs

y.value_counts()


# In[567]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)


# In[568]:


X.head()


# In[569]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components='mle'))  
        ]), ['Rating', 'age','years_married','religion','Education']),
        ('cat', OneHotEncoder(drop='first'), ['gender','children'])

    ])


# In[570]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


# In[571]:


model.fit(X_train,y_train)


# In[572]:


y_predict = model.predict(X_test)


# In[573]:


print(metrics.classification_report(y_test,y_predict)) 


# In[576]:


from sklearn.model_selection import GridSearchCV


# In[577]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])


# In[578]:


param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

model_grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',verbose=3)
model_grid.fit(X_train, y_train)


# In[579]:


best_model = model_grid.best_estimator_


# In[580]:


y_predict = best_model.predict(X_test)


# In[581]:


print(metrics.classification_report(y_test,y_predict))               #### we have the same issue


# In[587]:


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['gender','children']),
        ('num', StandardScaler(), ['Rating', 'age','years_married','religion','Education'])
    ])

#### lets see without PCA


# In[588]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RidgeClassifier(alpha=1.0))
])


# In[589]:


model.fit(X_train,y_train)


# In[590]:


y_predict = model.predict(X_test)


# In[591]:


metrics.accuracy_score(y_test,y_predict)


# In[592]:


print(metrics.classification_report(y_test,y_predict))      


# In[593]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


# In[594]:


model.fit(X_train,y_train)


# In[595]:


y_predict = model.predict(X_test)


# In[596]:


print(metrics.classification_report(y_test,y_predict))              #### seems like PCA in this case wasnt helping us


# In[597]:


from sklearn.model_selection import GridSearchCV


# In[598]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])


# In[599]:


param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

model_grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',verbose=3)
model_grid.fit(X_train, y_train)


# In[600]:


best_model = model_grid.best_estimator_


# In[601]:


y_predict = best_model.predict(X_test)


# In[602]:


print(metrics.classification_report(y_test,y_predict))               #### this is the best we have seen out of this


# In[ ]:


############################################################################################################################
#### We have decided to halt the current phase of our machine learning model development. Despite employing techniques #####
#### such as PCA, Standard Scaler, and handling class imbalance through various imbalanced pipelines, the model's ##########
#### performance has plateaued. The primary challenge lies in the limited dataset size, which hinders effective training.###
#### Additionally, the significant imbalance in the target variable further exacerbates the difficulty in achieving ########
#### meaningful improvements. ##############################################################################################
############################################################################################################################

