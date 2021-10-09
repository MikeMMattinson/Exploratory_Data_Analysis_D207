'''
########################################
    IMPORT REQUIRED PACKAGES
######################################## '''

# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import scipy.stats as stats
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from IPython.display import Latex
import sys
import os
import datetime
from mm_lib import *

# remove the ellipsis from terminal pandas tables
pd.set_option("display.width", None)
np.set_printoptions(threshold=np.inf)

# reconfigure sys.stdout to utf-8
sys.stdout.reconfigure(encoding="utf-8")

'''
########################################
    STUDENT INFORMATION
######################################## '''

# print student information
print(heading_toString("STUDENT INFORMATION"))
stud_info = {
  "Student": "Mike Mattinson",
  "Student ID": "001980761",
  "Class": "D207 Exploratory Data Analysis",
  "Dataset": "Churn",
  "Submission": "2nd Submission",
  "School": "WGU",
  "Instructor": "Dr. Sewell"
}
for key in stud_info:
    print('{:>14} | {}'.format(key,stud_info[key]))

print("Today is {}".format(datetime.date.today().strftime("%B %d, %Y")))


# print python enivronment
print(heading_toString("PYTHON ENVIRONMENT"))
print("Version: {} located at {}".format(sys.version, sys.executable))


# global variables
count_tables = 0  # keep track of the number of tables generated
count_figures = 0  # keep track of the number of figures generated
course = "d207"
fig_title_fontsize = 18
target = "Churn"
title_str = "CHURN DATA - WGU - D207 - MATTINSON 2021"

# check and create folder to hold all figures
figure_folder='figures\\' + course + '\\'
if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)

'''
########################################
    SETUP DATAFRAME
######################################## '''

# create dataframe
print(heading_toString("DATAFRAME (DF)"))
df = pd.read_csv("data\churn_clean.csv")
print(df[["Customer_id", "City", "Churn", "MonthlyCharge", "Tenure"]].head(4))
print(df.shape)

# remove unwanted columns
print(heading_toString("REMOVE UNWANTED COLUMNS"))

# remove unwanted columns
unwanted_columns = ["UID", "Interaction", "Lat", "Lng"]

for uc in unwanted_columns:
    if uc in df.columns:
        df.drop(columns=uc, inplace=True)
        print("[{}] column removed.".format(uc))

# show remaining columns
print("Remaining columns: \n{}".format(df.columns))


print(heading_toString("CONTINUOUS DATA"))
print(df.select_dtypes(include="float").info())

print(heading_toString("INTEGER DATA"))
print(df.select_dtypes(include="integer").info())

print(heading_toString("CATEGORICAL DATA"))
print(df.select_dtypes(include="object").info())

# output dataframe as .csv table
print(heading_toString("SAVE CLEAN DATAFRAME"))
count_tables += 1
table_title = "churn_clean_data"
fname = (
    "tables\\" + course + "\\" + "Tab_" + str(count_tables) + "_" + table_title + ".csv"
)
print("table saved at: {}".format(fname))
df.to_csv(fname, index=False, header=True)


'''
########################################
    PART B - CHI-SQUARE TEST
######################################## '''

# CHI-SQUARE TEST
target = 'Churn' 
prob = 0.999 
predictor = 'MonthlyCharge' # numerical
print(heading_toString("PART B - CHI-SQUARE INDEPENDENCE TEST - " + target + " vs. " + predictor))
bins = 6

# use library function to perform test
chi_square_analysis(target,predictor,bins,prob,df)

# use library function to print out distribution table
print(heading_toString("CHI-SQUARE DISTRIBUTION TABLE"))
print(chi_square_dist_table(0,0))

'''
########################################
    PART C - UNIVARIATE CATEOGRICAL
######################################## '''

print(heading_toString("PLOT UNIVARIATE CATEGORICAL"))
# box plot of Churn (cat)
#sns.countplot(df['InternetService'])
#plt.show()

# visualization of selected categorical data
univariate_categorical = {
  "1": "Churn",
  "2": "InternetService",
}

for key in univariate_categorical:
    fig, ax = plt.subplots()
    c = univariate_categorical[key]
    sns.countplot(df[c])
    plt.xlabel(c)
    plt.ylabel('Count')
    count_figures = key
    #title = 'Hello world'
    sub_title = '{}_{}_{}'.format(c,'categorical','countplot')
    title = '{}'.format(c)
    fname = 'Fig_{}_{}.{}'.format(str(count_figures), sub_title, "png")
    plt.title(title, fontsize=14, fontweight="bold")
    fig.suptitle(sub_title, fontsize=14)
    plt.figtext(0.5, 0.01, fname, ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    fig.tight_layout()

    # save file in the course figure's folder
    plt.savefig(os.path.join(figure_folder, fname))
    plt.close()
    print('figure saved at: [{}]'.format(fname))


'''
########################################
    PART C - UNIVARIATE CONTINUOUS
######################################## '''

print(heading_toString("DESCRIPTIVE STATS"))
# print descriptive stats for the dataframe
print(df.select_dtypes(include="float").describe().round(2))

print(heading_toString("PLOT UNIVARIATE CONTINUOUS"))

# visualization of selected univariate continuous data
univariate_continuous = {
  "3": "MonthlyCharge",
  "4": "Income",
}

for key in univariate_continuous:
    fig, ax = plt.subplots()
    c = univariate_continuous[key]
    plt.scatter(df.index,df[c])
    plt.xlabel(c)
    plt.ylabel('Count')
    count_figures = key
    #title = 'Hello world'
    sub_title = '{}_{}_{}'.format(c,'continuous','scatterplot')
    title = '{}'.format(c)
    fname = 'Fig_{}_{}.{}'.format(str(count_figures), sub_title, "png")
    plt.title(title, fontsize=14, fontweight="bold")
    fig.suptitle(sub_title, fontsize=14)
    plt.figtext(0.5, 0.01, fname, ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    fig.tight_layout()

    # save file in the course figure's folder
    plt.savefig(os.path.join(figure_folder, fname))
    plt.close()
    print('figure saved at: [{}]'.format(fname))




'''
########################################
    PART D - BIVARIATE 
######################################## '''
print(heading_toString("PLOT BIVARIATE COUNTPLOT"))
# visualization of selected bivariate data
bivariate = {
  '11': 'InternetService',
  '12': 'TechSupport',
  '13': 'PaymentMethod',
  '14': 'Tablet',
  '15': 'Gender',
  '16': 'Port_modem',
  '17': 'Techie',
}

for key in bivariate:
    fig, ax = plt.subplots()
    c = bivariate[key]
    sns.countplot(x='Churn', hue=c, data=df, palette='hls')
    plt.xlabel('{} vs. {}'.format(c,'Churn'))
    plt.ylabel('Count')
    count_figures = key
    sub_title = '{}_{}_{}'.format(c,'bivariate','countplot')
    title = '{} vs. {}'.format(c,'Churn')
    fname = 'Fig_{}_{}.{}'.format(str(count_figures), sub_title, "png")
    plt.title(title, fontsize=14, fontweight="bold")
    fig.suptitle(sub_title, fontsize=14)
    plt.figtext(0.5, 0.01, fname, ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    fig.tight_layout()

    # save file in the course figure's folder
    plt.savefig(os.path.join(figure_folder, fname))
    plt.close()
    print('figure saved at: [{}]'.format(fname))

print(heading_toString("PLOT BIVARIATE STACKED HISTOGRAM"))
# visualization of selected continous data
bivariate = {
  '18': 'MonthlyCharge',
  '19': 'Bandwidth_GB_Year',
  '20': 'Tenure',
  '21': 'Outage_sec_perweek',
  '22': 'Income',
}

for key in bivariate:
    c = bivariate[key] 
    target = 'Churn' 

    # print crosstab table associated with figure
    print(heading_toString(c + " crosstab"))
    print(pd.crosstab(pd.cut(df[c], bins = 6), df[target], margins=True))

    df_yes = df[df.Churn == "Yes"][c]
    df_no = df[df.Churn == "No"][c]
    yes_mean = df_yes.mean()
    no_mean = df_no.mean()
    fig, ax = plt.subplots()
    _, bins, _ = ax.hist([df_yes, df_no], bins=6, stacked=True)
    ax.legend(["Churn - Yes", "Churn - No"])
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ax.axvline(yes_mean, color="blue", lw=2)  # yes mean
    ax.axvline(no_mean, color="orangered", lw=2)  # no mean
    ax.text(
        (xmax - xmin) / 2,
        (ymax - ymin) / 2,
        "Delta:\n" + str(round(abs(yes_mean - no_mean), 2)),
        bbox={"facecolor": "white"},
    )
    plt.xlabel('{} vs. {}'.format(c,'Churn'))
    plt.ylabel('Count')
    count_figures = key
    sub_title = '{}_{}_{}'.format(c,'bivariate','stacked histogram')
    title = '{} vs. {}'.format(c,'Churn')
    fname = 'Fig_{}_{}.{}'.format(str(count_figures), sub_title, "png")
    plt.title(title, fontsize=14, fontweight="bold")
    fig.suptitle(sub_title, fontsize=14)
    plt.figtext(0.5, 0.01, fname, ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    fig.tight_layout()

    # save file in the course figure's folder
    plt.savefig(os.path.join(figure_folder, fname))
    plt.close()
    print('figure saved at: [{}]'.format(fname))



print(heading_toString("PLOT BIVARIATE STACKED BAR"))
# visualization of selected bivariate data
bivariate = {
    "23": "Gender",
    "24": "InternetService",
}

for key in bivariate:
    c = bivariate[key]
    target = 'Churn'

    # print crosstab table associated with figure
    print(heading_toString(c + " crosstab"))
    print(pd.crosstab(df[c], df[target], margins=True))

    y = DataFrame({"count": df.groupby([c, target]).size()}).reset_index()

    """ each dataframe will look like this:
    Phone Churn  count
0    No    No   1351
1    No   Yes    521
2   Yes    No   5999
3   Yes   Yes   2129
    """

    x = y[c].unique()

    fig, ax = plt.subplots()
    no = y[y[target] == "No"]
    yes = y[y[target] == "Yes"]

    ax.barh(
        x,
        yes["count"],
        height=0.75,
        color="lightgreen",
        label="Churn Yes",
        left=no["count"],
    )
    ax.barh(x, no["count"], height=0.75, color="darkgreen", label="Churn No")
    ax.legend(["Churn - Yes", "Churn - No"])
    plt.xlabel('{} vs. {}'.format(c,'Churn'))
    plt.ylabel('Count')
    count_figures = key
    sub_title = '{}_{}_{}'.format(c,'bivariate','stacked bar')
    title = '{} vs. {}'.format(c,'Churn')
    fname = 'Fig_{}_{}.{}'.format(str(count_figures), sub_title, "png")
    plt.title(title, fontsize=14, fontweight="bold")
    fig.suptitle(sub_title, fontsize=14)
    plt.figtext(0.5, 0.01, fname, ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    fig.tight_layout()

    # save file in the course figure's folder
    plt.savefig(os.path.join(figure_folder, fname))
    plt.close()
    print('figure saved at: [{}]'.format(fname))

   
