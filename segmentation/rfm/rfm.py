import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH")

class RFM:
    def __init__(self):
        self.RFM = pd.DataFrame()
        self.RFM_outlier_free = pd.DataFrame() 
        self.df = pd.read_excel(DATASET_PATH)
        print(self.df.info())
    
    def data_prep(self):
        self.df.dropna(inplace=True)
        self.df = self.df[~self.df["Invoice"].str.contains("C", na=False)]
        self.df['TotalPrice'] = self.df['Quantity'] * self.df['Price']
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        self.df = self.df[self.df['Country'] == 'United Kingdom']
        print(self.df.head())
    
    def eda(self):
        pass
    
    def get_rfm_values(self):
        today_date = self.df['InvoiceDate'].max() + dt.timedelta(days=2)
        self.RFM = self.df.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                    'Invoice': lambda inv: inv.nunique(),
                                    'TotalPrice': lambda price: price.sum()})
        self.RFM.columns=['Recency','Frequency','Monetary']
        self.RFM['Recency'] = pd.to_numeric(self.RFM['Recency'], errors='coerce')
        self.RFM['Frequency'] = pd.to_numeric(self.RFM['Frequency'], errors='coerce')
        self.RFM['Monetary'] = pd.to_numeric(self.RFM['Monetary'], errors='coerce')
        self.RFM = self.RFM.dropna()
        self.RFM['Recency'] = self.RFM['Recency'].astype(int)
        self.RFM['Frequency'] = self.RFM['Frequency'].astype(int)
        self.RFM['Monetary'] = self.RFM['Monetary'].astype(float)
        print(self.RFM.head())
    
    def calculate_rfm_score(self):
        self.RFM["R"] = pd.qcut(self.RFM['Recency'], 5, labels=[5,4,3,2,1])
        self.RFM["F"] = pd.qcut(self.RFM['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        self.RFM["M"] = pd.qcut(self.RFM['Monetary'], 5, labels=[1,2,3,4,5])
        self.RFM['RFM'] = self.RFM['R'].astype(str) + self.RFM['F'].astype(str) + self.RFM['M'].astype(str)
        self.RFM['RFM']   = self.RFM['RFM'].astype(int)
        self.RFM['R']   = self.RFM['R'].astype(int)
        self.RFM['F']   = self.RFM['F'].astype(int)
        self.RFM['M']   = self.RFM['M'].astype(int)

        print(self.RFM.head())
        print(self.RFM.info())

    def remove_outliers(self):
        Q1 = self.RFM.quantile(0.25)
        Q3 = self.RFM.quantile(0.75)
        IQR = Q3 - Q1
        #print('Q1\n',Q1,'\nQ3\n',Q3,'\nIQR\n',IQR)
        self.RFM_outlier_free = self.RFM[(self.RFM['Recency'] > (Q1['Recency'] - 1.5 * IQR['Recency']))&(self.RFM['Recency'] < (Q3['Recency'] + 1.5 * IQR['Recency']))]
        self.RFM_outlier_free = self.RFM_outlier_free[(self.RFM_outlier_free['Frequency'] > (Q1['Frequency'] - 1.5 * IQR['Frequency']))&(self.RFM_outlier_free['Frequency'] < (Q3['Frequency'] + 1.5 * IQR['Frequency']))]
        self.RFM_outlier_free = self.RFM_outlier_free[(self.RFM_outlier_free['Monetary'] > (Q1['Monetary'] - 1.5 * IQR['Monetary']))&(self.RFM_outlier_free['Monetary'] < (Q3['Monetary'] + 1.5 * IQR['Monetary']))]
        print("Number of outliers removed  = ",len(self.RFM)-len(self.RFM_outlier_free))

        print(self.RFM_outlier_free.head())
        print(self.RFM_outlier_free.info())
    
    def plot_boxplot(self):
        sns.boxplot(x=self.RFM_outlier_free['Recency'],color='red',showfliers=False)
        plt.show()
        sns.boxplot(x=self.RFM_outlier_free['Frequency'],color='Blue',showfliers=False)
        plt.show()
        sns.boxplot(x=self.RFM_outlier_free['Monetary'],color='Green',showfliers=False)
        plt.show()