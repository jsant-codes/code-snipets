#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Environment:
import numpy as np
import pandas as pd
import date_time

#for webscraping
import time
from bs4 import BeautifulSoup
import requests as req

#for plots
import seaborn as sns
import matplotlib.pyplot as plt
#for stats
import numpy as np
from scipy import stats

#pandas environment
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 100)

#initial review code
display(df.head(10))
print(round(df.describe()))
print('\n')
print(df.dtypes)


#missing data by prct
maxRows = df['id column'].count()
 
print('% Missing Data:')
print((1 - df.count() / maxRows) * 100)


# In[1]:


###Pandas Functions
#reading and writing pandas files
df = pd.read_csv ("C:\Users\jonat\OneDrive\Desktop\data\python_csvs")
df.to_csv(file)

#descriptives
df.info()
df.describe()
display(df.head())
df.dtypes #kind of useless just use info at start then dtypes if you only want coumn type
df.column.unique()

#slicing
    #one col
        slice_1 = df['column1']
    #multiple col
         slice_2 = df[['column1', 'column2']]
        
# selecting based on two attibutes
selection = df[(df.column_1 == 'criteria 1') & (df.column_2 == 'criteria 2')]
        
#slice rows
    examples = df.iloc[2]
#slice multiple rows
    example_rows = df.iloc[3:6]
#slice row with logic
    df[df.example > 30]
#slice row with multiple OR indices
    march_april = df[(df.month == 'March') | (df.month == 'April')]
#slice rows with 
    january_february_march = df[df.month.isin(['January', 'February', 'March'])]
#slice rows with & logic 
    frances_palmer = orders[(orders.first_name == 'Frances') & (orders.last_name == 'Palmer')]
#index reset when subsetting need to do to cleanup
    df3 = df2.reset_index(inplace=True, drop=True)

#subset df
df_female = df[df.sex == 'female']
    

#lambda format for row ops
df['column'] = df.apply(lambda row:
    row['column'] * int
    if row['column_2'] == 'Yes'
    else row['column_3'],
    axis=1
)

#lambda column ops
get_last_name = lambda x: x.split()[-1]  
df['last_name'] = df.name.apply(get_last_name)

##lambda rows multi column input
total_earned = lambda row: (row.hourly_wage * 40) + ((row.hourly_wage * 1.5) * (row.hours_worked - 40)) if row.hours_worked > 40 else row.hourly_wage * row.hours_worked
  
df['total_earned'] = df.apply(total_earned, axis = 1)

#splitting column
df_split = df['column unsplit'].str.split(',')
df['column_split_1'] = df_split.str.get(0)
df['column_split_2'] = df_split.str.get(1)
df['column_split_3'] = df_split.str.get(2)
df = df.pop('column unsplit')

#split one column into more
students['gender']=students.gender_age.str[0]
students['age']=students.gender_age.str[1:3]

#small lambdas
US = wine['country'].map(lambda x: str(x).startswith('US'))
ends_in_a = lambda str: "a" in str[-1]
is_substring = lambda my_string: my_string in "This is the master string"
check_if_A_grade = lambda grade: 'Got an A!' if grade >= 90 else 'Did not get an A...'
double_or_zero = lambda num: 2*num if num > 10 else 0
even_or_odd = lambda num: "even" if num%2 == 0 else "odd"
#two solutliosn for ones place
ones_place = lambda num: str(num)[-1]
ones_place = lambda num: num%10

#applying function
    df['name'] = df.apply(function, axis = 1)

###Cleaning
    #sort vals very important for filling
        df = df.sort_values(['RespondentID','Year'])

    #drop duplicates
        df.drop_duplicated()
        print(df.shape())
        #can also use is_duplicated() i think
        #reshape to confirm df.shape also maybe reset_index
        
    #dropna listwise deletion drops all rows with missing 
    df.dropna(inplace=True)
    #dropna pairwise maintains more data
        df.dropna(subset=['Height','Education'], #only looks at these two columns
                inplace=True, #removes the rows and keeps the data variable
                how='any') #removes data with missing data in either field

    #work with all columns
        example.columns = map(str.lower, example.columns)
    #rename individual columns
        example = example.rename({'dba': 'name', 'cuisine description': 'cuisine'}, axis=1)
        
    #drop individual columns delete column
        df.drop(['column1','column2','column3'],
        axis=1,
        inplace=True)
    
    # view unique values for each column
        df.nunique() 
    #see specific NA vlaues
        df.isna().sum() 
    #fillna
    df = df.fillna(value=variable_name, inplace=True) #fills all null vaues with value given
    
    #forward fill for using 
    df['comfort'].ffill(axis=0, inplace=True)
    #option bfill
    
    #replace instances of value by column
        df['example'] = df['example'].where(df['example'] < 40, np.nan)
    #cross tab 
    pd.crosstab(

            # tabulates the boroughs as the index
            restaurants['boro'],  

            # tabulates the number of missing values in the url column as columns
            restaurants['url'].isna(), 

            # names the rows
            rownames = ['boro'],

            # names the columns 
            colnames = ['url is na']) 

    #strip column strings
        df['url'] = df['url'].str.lstrip('https://') 

    #.str.lstrip('www.') removes the “www.” from the left side of the string
        df['url'] = df['url'].str.lstrip('www.') 

    #melt values
    annual_wage=annual_wage.melt(
          # which column to use as identifier variables
          id_vars=["boro"], 

          # column name to use for “variable” names/column headers (ie. 2000 and 2007) 
          var_name=["year"], 

          # column name for the values originally in the columns 2000 and 2007
          value_name="avg_annual_wage") 
    df = df.melt(
        id_vars=['name']
        var_name=['name_2']
        value_name='value'
    )
    
#.get() function + for dictionary search plus can specify result if no vlaue is found
tc_id = user_ids.get('teraCoder', 100000)

spread = {}
spread['past'] = tarot.pop(13)
spread['present'] = tarot.pop(22)
spread['future'] = tarot.pop(10)

#pandas function
inventory['instock'] = inventory['quantity'].apply(lambda row: True if row > 0 else False)

inventory['total_value'] = inventory['price'] * inventory['quantity']

combine_lambda = lambda row:     '{} - {}'.format(row.product_type,
                     row.product_description)

inventory['full_description'] = inventory.apply(combine_lambda, axis = 1)

high_earners = df.groupby('category').wage
    .apply(lambda x: np.percentile(x, 75))
    .reset_index()
    
shoe_counts = orders.groupby(['shoe_type', 'shoe_color']).id.count().reset_index()


# In[ ]:


#python function
toomer_bio_fixed = toomer_bio.replace('Tomer', 'Toomer')




# In[ ]:


#Stats testing
    #shapiro test for normality of data
    #interpretation: if p result is greater than .05 then results are significant aka normal
    #needs num<2000 and continous variable
normality = stats.shapiro(df.column)

    #Pearson R
    #interpretation:
    #corr: tells relationshiop. closer to 1 or -1 then stronger linear relationship can also use log regression if data not normal but normalizes in log curve
    #pval: if >.05 then stat sig
corr_smoker_charges, p = stats.pearsonr(df.smoker_binary, df.charges)
    #maybe can do CIs 
    corr_smoker_charges.confidence_interval

    #covariance
    #interpretaion: 
    cov_var1_var2 = np.cov(df.var1, df.var2)
    


# In[ ]:


###Matplot Plts
    #histogram
plt.hist(df.charges)
plt.title('Histogram of Charges')
plt.xlabel('Charge in USD')
plt.ylabel('Count')
plt.show()
plt.close()
#plt.clf() (if showing multiple plots in same output)
#two plot overlayed hist where alpha is the opacity variable
plt.hist(scores_urban.column, color='blue', label='Urban', normed=True, alpha=0.5)
plt.hist(scores_rural.column, color='red', label='Rural', normed=True, alpha=0.5)
plt.legend(list to use, loc=??)
plt.title('xxx')

ax= plt.sublplot()
plt.plot()
ax.set_yticks
ax.set_yticks([0.1, 0.6, 0.8])
ax.set_yticklabels(['10%', '60%', '80%'])

#plot legend loc = -->
Number Code	String
0	best
1	upper right
2	upper left
3	lower left
4	lower right
5	right
6	center left
7	center right
8	lower center
9	upper center
10	center


df['quality'].hist(bins=20);

    #scatter plot
plt.scatter(x = housing.beds, y = housing.sqfeet)
plt.xlabel('Number of beds')
plt.ylabel('Number of sqfeet')


###Seaborn Plots

    #Boxplot
sns.boxplot(data=nba_2014, x='pts', y='fran_id')
    #Bar Graph
sns.countplot(x='genre', data=movies)
    #pie chart
movies.genre.value_counts().plot.pie()

#bar graph
A=sns.catplot(
    data=missingData, kind="bar",
    x="Country", y="Employment",
    height = 6, aspect = 2)
B=sns.catplot(
    data=missingData, kind="bar",
    x="Country", y="DevType",
    height = 6, aspect = 2)

#show missing data
missingUndergrad = df['UndergradMajor'].isnull().groupby(df['Year']).sum().reset_index()
 
sns.catplot(x="Year", y="UndergradMajor",
                data=missingUndergrad, kind="bar",
                height=4, aspect=1);

sns.regplot(x='points', y='price', x_estimator=np.mean, data=wine)

#line graphs
plt.plot(x,y)
#where x is list and y is list of equal value count

#change color and lineestyle and marker
plt.plot(time, revenue, color='purple', linestyle = '--')
plt.show()
plt.close()
plt.plot(time,costs, color = '#82edc9', marker = 's')
plt.show()


#subplots 
# plt.sublplot (rows, columns, index of plot ie. first in list, second etc.)
plt.subplot(1, 2, 1)
plt.plot(months, temperature)
plt.title('Temp vs Months')

plt.subplot(1,2,2)
plt.plot(temperature, flights_to_hawaii)
plt.title('temp vs flights')
plt.show()

#adjust margins
plt.subplots_adjust(wspace=.35, bottom=.2)
left — the left-side margin, with a default of 0.125. You can increase this number to make room for a y-axis label
right — the right-side margin, with a default of 0.9. You can increase this to make more room for the figure, or decrease it to make room for a legend
bottom — the bottom margin, with a default of 0.1. You can increase this to make room for tick mark labels or an x-axis label
top — the top margin, with a default of 0.9
wspace — the horizontal space between adjacent subplots, with a default of 0.2
hspace — the vertical space between adjacent subplots, with a default of 0.2



# When to use charts:
# Temporal changes: Line chart, area charts
# Parts of a whole: Pie chart, Treemap
# Relationship between variables: Scatterplot
# Distribution: Histogram, box plot
# Ranking or Magnitude: Bar chart, packed bubbles
