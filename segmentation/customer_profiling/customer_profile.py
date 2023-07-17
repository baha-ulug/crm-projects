import pandas as pd
import numpy as np
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns


def gen_df():
    # Veri setinin boyutunu belirleme
    num_customers = 1000

    # Rastgele müşteri ID'leri oluşturma
    customer_ids = range(1, num_customers + 1)

    # Rastgele yaşlar oluşturma
    ages = np.random.randint(18, 65, size=num_customers)

    # Rastgele gelir seviyeleri oluşturma
    income_levels = np.random.choice(['Low', 'Medium', 'High'], size=num_customers, p=[0.4, 0.4, 0.2])

    # Rastgele ilgi alanları oluşturma
    interests = np.random.choice(['Sports', 'Music', 'Travel', 'Fashion'], size=num_customers)

    # Rastgele satın alma alışkanlıkları oluşturma
    purchase_freq = np.random.randint(1, 10, size=num_customers)

    # Müşteri profili veri setini oluşturma
    customer_data = {
        'CustomerID': customer_ids,
        'Age': ages,
        'IncomeLevel': income_levels,
        'Interest': interests,
        'PurchaseFrequency': purchase_freq
    }

    df = pd.DataFrame(customer_data)
    return df

df = gen_df()


def plot_df(df):
    # İlk 5 satırı gösterme
    print(df.head())

    # Müşteri yaş dağılımını görselleştirme
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='Age', bins=10, kde=True)
    plt.title('Age Distribution of Customers')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()

    # Gelir seviyesi dağılımını görselleştirme
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='IncomeLevel')
    plt.title('Income Level Distribution of Customers')
    plt.xlabel('Income Level')
    plt.ylabel('Count')
    plt.show()

plot_df(df)


def segmentation(df):
    # Yaş ve gelir seviyesine göre segmentasyon yapma
    df['age_segment'] = pd.cut(df['age'], bins=[15, 25, 35, 45, 55, 65], labels=['15-25', '26-35', '36-45', '46-55', '56-65'])
    df['income_segment'] = pd.cut(df['income_level'], bins=['Low', 'Medium', 'High'], labels=['Low', 'Medium', 'High'])

    # Segmentlere göre gruplama ve istatistiklerin hesaplanması
    segment_stats = df.groupby(['age_segment', 'income_segment']).agg({
        'uid': 'count',
        'reg_date': ['min', 'max'],
        'gender': lambda x: x.value_counts().index[0],
        'country': lambda x: x.value_counts().index[0]
    }).reset_index()

    # Sonuçları yazdırma
    print(segment_stats)

def profiling(df):
    # Gelir seviyesine göre gruplama ve ortalama satın alma sıklığı analizi
    income_segment_analysis = df.groupby('IncomeLevel').agg({'PurchaseFrequency': 'mean'}).reset_index()
    print(income_segment_analysis)

    # İlgi alanına göre gruplama ve ilgi alanlarının yüzdesel dağılımı analizi
    interest_analysis = df['Interest'].value_counts(normalize=True).reset_index()
    interest_analysis.columns = ['Interest', 'Percentage']
    print(interest_analysis)