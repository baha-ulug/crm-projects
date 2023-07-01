############################################
# ASSOCIATION_RULE_LEARNING
############################################

############################################
# Data Preparation
###########################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
from dotenv import load_dotenv
from preprocess.eda import check_df
from preprocess.data_prep import crm_data_prep, create_invoice_product_df
import os 

def read_data():
    #dotenv_path = os.path.join(os.getcwd(), '.env')
    #load_dotenv(dotenv_path, verbose=True)
    #load_dotenv()
    #DATASET_PATH = os.getenv("DATASET_PATH")
    DATASET_PATH = "C:/Users/baha.ulug/Desktop/projects/crm-projects/datasets/online_retail_II.xlsx"
    df = pd.read_excel(DATASET_PATH)
    return df

def data_prep(df):
    df.dropna(inplace=True)
    df = df[~df["Invoice"].str.contains("C", na=False)]
    df['TotalPrice'] = df['Quantity'] * df['Price']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

def eda_df():
    #Her bir ürünün StockCode a göre Quantity sini sum ediyoruz.
    df.groupby(["Invoice","StockCode"]).agg({"Quantity":"sum"}).head(10)

    #Invoice ları tekillestiriyoruz.
    df.groupby(["Invoice","StockCode"]).agg({"Quantity":"sum"}).unstack().iloc[0:5,0:5]

    df[(df["StockCode"] == 16235) & (df["Invoice"] == 538174)]

    #Satırlarda sadece bir adet fatura adı olsun.
    #Sütunlarda ürünler olsun.
    #kesişiminde hangi faturalardan kaçar tane olduğu yazsın.
    df.groupby(["Invoice","StockCode"]).\
        agg({"Quantity":"sum"}).\
        unstack().fillna(0).iloc[0:5,0:5]

    #Veriyi beklenen product forma getirdik.
    df.groupby(["Invoice","StockCode"]).\
        agg({"Quantity":"sum"}).\
        unstack().fillna(0).\
        applymap(lambda i:1 if i>0 else 0).iloc[0:5,0:5]

    # Her bir invoice'da kaç eşsiz ürün vardır.
    df.groupby("Invoice").agg({"StockCode":"nunique"})

    # Her bir product kaç eşsiz sepettedir.
    df.groupby("StockCode").agg({"Invoice":"nunique"})

############################################
# ASSOCIATION_RULE
############################################
#Use apriori.
#Select min_support value is 0.01.
#Use_colnames= use column names.
#Select supports of combinations of each item.

#conviction: Y olmadan X in beklenen frekansıdır.
#leverage: Lifte benzer. Supportu yüksek değere öncelik verir. Yanlıdır.
#lift : Daha az sıklığa rağmen güçlü ilişkileri bulabilir. Yansızdır. Support düşük olsa da ilişkiler nettir.

# Stockcode yerine ürün isimlerini aldık.
def create_invoice_product_df(dataframe):
    return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
        applymap(lambda x: 1 if x > 0 else 0)

def create_rules(dataframe, country=False, head=5):
    if country:
        dataframe = dataframe[dataframe['Country'] == country]
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))
    else:
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))

    return rules

df = read_data()
check_df(df)
df=data_prep(df)
rules = create_rules(df,"Germany")

#order by support and lift.
rules.sort_values(["support","lift"], ascending= [False,False]).head()