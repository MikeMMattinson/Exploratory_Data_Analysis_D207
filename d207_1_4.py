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
import datetime
from mm_lib import *

# remove the ellipsis from terminal pandas tables
pd.set_option("display.width", None)
np.set_printoptions(threshold=np.inf)

# reconfigure sys.stdout to utf-8
sys.stdout.reconfigure(encoding="utf-8")

# print student information
print(heading_toString("STUDENT INFO - 2ND SUBMISSION"))
print("Student: {}".format("Mike Mattinson"))
print("Student ID: {}".format("001980761"))
print("Class: {}".format("D207 Exploratory Data Analysis"))
print("Dataset: {}".format("Churn"))
print("School: {}".format("WGU"))
print("Instructor: {}".format("Dr. Sewell"))
print("Mentor: {}".format("Fiki Revels"))
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


# create dataframe
print(heading_toString("DATAFRAME (DF)"))
df = pd.read_csv("data\churn_clean.csv")
print(df[["Customer_id", "City", "Churn", "MonthlyCharge", "Tenure"]].head(4))
print(df.shape)
print(df.info())


# remove unwanted columns
print(heading_toString("REMOVE UNWANTED COLUMNS"))

# remove the following unwanted columns
unwanted_columns = ["UID", "Interaction", "Lat", "Lng"]

for uc in unwanted_columns:
    if uc in df.columns:
        df.drop(columns=uc, inplace=True)
        print("[{}] column removed.".format(uc))

# show remaining columns
print("Remaining columns: \n{}".format(df.columns))


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
╔═╗╔═╗╦═╗╔╦╗  ╔╗   ╔═╗╦ ╦╦   ╔═╗╔═╗ ╦ ╦╔═╗╦═╗╔═╗╔╦╗  ╔╦╗╔═╗╔═╗╔╦╗
╠═╝╠═╣╠╦╝ ║   ╠╩╗  ║  ╠═╣║───╚═╗║═╬╗║ ║╠═╣╠╦╝║╣  ║║   ║ ║╣ ╚═╗ ║ 
╩  ╩ ╩╩╚═ ╩   ╚═╝  ╚═╝╩ ╩╩   ╚═╝╚═╝╚╚═╝╩ ╩╩╚═╚═╝═╩╝   ╩ ╚═╝╚═╝ ╩ 
'''

# CHI SQUARE DISTRIBUTION TABLE
print(heading_toString("CHI-SQUARE DISTRIBUTION TABLE"))
print(chi_square_dist_table(0,0))


# CHI^2 INDEPENDENCE TEST
target = 'Churn' # used for all of the following chi-square tests
prob = 0.999 # used for all of the following chi-square tests
predictor = 'MonthlyCharge' # numerical
print(heading_toString("CHI^2 INDEPENDENCE TEST - " + target + " vs. " + predictor))
bins = 6
chi_square_analysis(target,predictor,bins,prob,df)





'''
╔═╗╔═╗╦═╗╔╦╗  ╔═╗       ╦ ╦╔╗╔╦╦  ╦╔═╗╦═╗╦╔═╗╔╦╗╔═╗
╠═╝╠═╣╠╦╝ ║   ║    ───  ║ ║║║║║╚╗╔╝╠═╣╠╦╝║╠═╣ ║ ║╣ 
╩  ╩ ╩╩╚═ ╩   ╚═╝       ╚═╝╝╚╝╩ ╚╝ ╩ ╩╩╚═╩╩ ╩ ╩ ╚═╝
╔═╗╔═╗╔╗╔╦╗╦╔╗╔╦ ╦╔═╗╦ ╦╔═╗                        
║  ║ ║║║║║ ║║║║║ ║║ ║║ ║╚═╗                        
╚═╝╚═╝╝╚╝╩ ╩╝╚╝╚═╝╚═╝╚═╝╚═╝                        
'''






'''
╔═╗╔═╗╦═╗╔╦╗  ╔╦╗       ╔╗ ╦╦  ╦╔═╗╦═╗╦╔═╗╔╦╗╔═╗
╠═╝╠═╣╠╦╝ ║    ║║  ───  ╠╩╗║╚╗╔╝╠═╣╠╦╝║╠═╣ ║ ║╣ 
╩  ╩ ╩╩╚═ ╩   ═╩╝       ╚═╝╩ ╚╝ ╩ ╩╩╚═╩╩ ╩ ╩ ╚═╝
╔═╗╔═╗╔╗╔╦╗╦╔╗╔╦ ╦╔═╗╦ ╦╔═╗                     
║  ║ ║║║║║ ║║║║║ ║║ ║║ ║╚═╗                     
╚═╝╚═╝╝╚╝╩ ╩╝╚╝╚═╝╚═╝╚═╝╚═╝                     
'''

# visualize selected numerical data
print(heading_toString("PART D - BIVARIATE CONTINUOUS"))


# visualization of selected continous data
selected_continous_columns = [
    "MonthlyCharge",
    "Bandwidth_GB_Year",
    "Tenure",
    "Outage_sec_perweek",
    "Income",
]

for c in selected_continous_columns:
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
    plt.xlabel(c)
    plt.ylabel('Churn')
    count_figures += 1
    title = fig_title_toString(c + " Data", count_figures, "hist")
    fig.suptitle(title_str, fontsize=14, fontweight="bold")
    fname = fig_fname_toString(course, title, "png")
    plt.title(title, fontsize=fig_title_fontsize)
    fig.tight_layout()
    plt.savefig(fname)
    plt.close()
    print("figure saved at: [{}]".format(fname))

print(heading_toString("DESCRIBE SELECTED NUMERICAL DATA"))
print(df[selected_numeric_columns].describe().round(2).T)  # show descriptive stats



'''
╔═╗╔═╗╦═╗╔╦╗  ╔═╗       ╦ ╦╔╗╔╦╦  ╦╔═╗╦═╗╦╔═╗╔╦╗╔═╗
╠═╝╠═╣╠╦╝ ║   ║    ───  ║ ║║║║║╚╗╔╝╠═╣╠╦╝║╠═╣ ║ ║╣ 
╩  ╩ ╩╩╚═ ╩   ╚═╝       ╚═╝╝╚╝╩ ╚╝ ╩ ╩╩╚═╩╩ ╩ ╩ ╚═╝
╔═╗╔═╗╔╦╗╔═╗╔═╗╔═╗╦═╗╦╔═╗╔═╗╦                      
║  ╠═╣ ║ ║╣ ║ ╦║ ║╠╦╝║║  ╠═╣║                      
╚═╝╩ ╩ ╩ ╚═╝╚═╝╚═╝╩╚═╩╚═╝╩ ╩╩═╝                    
'''


# visualize selected categorical data
print(heading_toString("VISUALIZATION OF SELECTED CATEGORICAL DATA"))

# print list of data's categorical values
print(df.select_dtypes(include="object").columns)

# visualization of selected numeric data
selected_categorical_columns = [
    "Gender",
    "Area",
    "InternetService",
    "Techie",
    "Tablet",
    "PaperlessBilling",
    "PaymentMethod",
]

for c in selected_categorical_columns:

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

    ax.barh(x, no["count"], height=0.75, color="darkgreen", label="Churn No")
    ax.barh(
        x,
        yes["count"],
        height=0.75,
        color="lightgreen",
        label="Churn Yes",
        left=no["count"],
    )

    fig.suptitle(title_str, fontsize=14, fontweight="bold")

    plt.xlabel("Count")
    plt.ylabel(c)
    _, xmax = plt.xlim()
    plt.xlim(0, xmax + 300)

    # legend and grid
    ax.grid(True)
    ax.legend()

    # create and save figure image
    count_figures += 1
    title = fig_title_toString(c + " Data", count_figures, "barh")
    fname = fig_fname_toString(course, title, "png")
    plt.title(title, fontsize=fig_title_fontsize)
    fig.tight_layout()
    plt.savefig(fname)
    plt.close()
    print("figure saved at: [{}]".format(fname))





# visualize bivariate numerical data
print(heading_toString("VISUALIZATION OF BIVARIATE DATA - Churn vs Income"))

# create scatterplot of Churn vs. MonthlyCharge
title = "Bivariate - Churn vs InternetService"
fig, ax = plt.subplots()
#ax1 = sns.catplot(x="Churn", y="Income", order=["No", "Yes"], data=df)
#ax1 = sns.barplot(x ='Churn', y ='Income', data = df, palette ='plasma')
#ax1 = sns.jointplot(x="Income", y="MonthlyCharge", data=df, kind="scatter")
#ax1 = sns.lmplot(
#    "Income", "Tenure", data=df, hue="Churn", fit_reg=False)
ax1 = sns.countplot(x='Churn', hue="InternetService", data=df, palette='hls')
fig.suptitle(title_str, fontsize=14, fontweight="bold")

# create and save figure image
count_figures += 1
title = fig_title_toString(title, count_figures, "countplot")
fname = fig_fname_toString(course, title, "png")
plt.title(title, fontsize=fig_title_fontsize)
fig.tight_layout()
plt.savefig(fname)
plt.close()
print("figure saved at: [{}]".format(fname))


# visualize bivariate numerical data
print(heading_toString("VISUALIZATION OF BIVARIATE DATA - Churn vs Tenure"))

# create scatterplot of Churn vs. MonthlyCharge
title = "Bivariate - Churn vs Tenure"
fig, ax = plt.subplots()
ax1 = sns.catplot(x="Churn", y="Tenure", order=["No", "Yes"], data=df)
fig.suptitle(title_str, fontsize=14, fontweight="bold")

# create and save figure image
count_figures += 1
title = fig_title_toString(title, count_figures, "scatter")
fname = fig_fname_toString(course, title, "png")
plt.title(title, fontsize=fig_title_fontsize)
fig.tight_layout()
plt.savefig(fname)
plt.close()
print("figure saved at: [{}]".format(fname))



