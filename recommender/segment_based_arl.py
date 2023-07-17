##############################################################
# CLTV Prediction with BG-NBD ve Gamma-Gamma 
##############################################################

##############################################################
# 1. Data Preperation
##############################################################

from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules
from lifetimes.plotting import plot_period_transactions
from dotenv import load_dotenv
import os 

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def read_data():
    #df =pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
    load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    df = pd.read_excel(DATASET_PATH)
    return df

def get_info(df):
    print(df.describe([0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)

def prep_data(df):
    df.dropna(inplace=True)
    df = df[~df["Invoice"].str.contains("C", na=False)]
    df = df[df["Quantity"] > 0]
    replace_with_thresholds(df, "Quantity")
    replace_with_thresholds(df, "Price")
    df["TotalPrice"] = df["Price"] * df["Quantity"]
    # UK müşterilerini seçme
    df = df[df["Country"] == "United Kingdom"]
    return df

def get_analyse_date(df):
    max_date = df["InvoiceDate"].max()
    today_date = max_date + timedelta(days=2)
    #today_date = dt.datetime(2011, 12, 11)
    return today_date

#############################################
# RFM Table
#############################################
# metrikleri oluşturma

def rfm_df(df):
    today_date = get_analyse_date(df)
    rfm = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max()-date.min()).days,
                                                        lambda date: (today_date-date.min()).days],
                                        'Invoice': lambda num: num.nunique(),
                                        'TotalPrice': lambda price: price.sum()})
    rfm.columns = rfm.columns.droplevel(0)

    # sütunları isimlendirme
    rfm.columns = ['recency_cltv_p', 'tenure', 'frequency', 'monetary']

    # monetary avg hesaplama --> Gamma Gamma modeli bu şekilde istiyor
    # We need to calculate the average profit:
    rfm["monetary"] = rfm["monetary"] / rfm["frequency"]

    rfm.rename(columns={"monetary": "monetary_avg"}, inplace=True)

    # recency ve tenure değişkenlerini haftalığa çevirme
    # BG/NBD model asks us for recency and T weekly
    rfm["recency_weekly_p"] = rfm["recency_cltv_p"] / 7
    rfm["tenure_weekly_p"] = rfm["tenure"] / 7

    # kontroller
    rfm = rfm[rfm["monetary_avg"] > 0]
    rfm = rfm[rfm["frequency"] > 1]
    rfm["frequency"] = rfm["frequency"].astype(int)
    return rfm

##############################################################
# 2. Creation of BG/NBD model
##############################################################

def fit_bgf(rfm):
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(rfm['frequency'],
            rfm['recency_weekly_p'],
            rfm['tenure_weekly_p'])
    return bgf

def pred_bgf(bgf,rfm,week=24,n_cust=10):
    # week/4 ay içinde en çok satın alma beklediğimiz n_cust müşteri kimdir?
    top_customers = bgf.conditional_expected_number_of_purchases_up_to_time(week,
                                                            rfm['frequency'],
                                                            rfm['recency_weekly_p'],
                                                            rfm['tenure_weekly_p']).sort_values(ascending=False)
    print(top_customers.head(n_cust))

def exp_sales(bgf,rfm,week=24):
    #Who are the 10 customers we expect the most to purchase in a month?
    rfm["exp_sales_6_month"] = bgf.predict(week,
                                            rfm['frequency'],
                                            rfm['recency_weekly_p'],
                                            rfm['tenure_weekly_p'])
    rfm.sort_values("exp_sales_6_month", ascending=False).head(20)
    return rfm

def expected_transaction(bgf,rfm,week=24):
    # 6 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?  952.4548865072431
    trasaction_count = bgf.predict(week,
                rfm['frequency'],
                rfm['recency_weekly_p'],
                rfm['tenure_weekly_p']).sum()
    print(f"Number of transaction in {week} week is: {trasaction_count}")

def eval_predictions(bgf):
    # Tahmin Sonuçlarının Değerlendirilmesi
    plot_period_transactions(bgf)
    plt.show()

##############################################################
# 3. Creation of GAMMA-GAMMA Model
##############################################################
def fit_ggf(rfm):
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(rfm['frequency'], rfm['monetary_avg'])
    return ggf

def pred_ggf(ggf,rfm):
    ggf.conditional_expected_average_profit(rfm['frequency'],
                                            rfm['monetary_avg']).sort_values(ascending=False).head(10)
    rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm['frequency'],
                                                                            rfm['monetary_avg'])
    return rfm

##############################################################
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################

def calculate_clv(bgf,ggf,rfm,month):
    cltv = ggf.customer_lifetime_value(bgf,
                                    rfm['frequency'],
                                    rfm['recency_weekly_p'],
                                    rfm['tenure_weekly_p'],
                                    rfm['monetary_avg'],
                                    time=month,
                                    freq="W",
                                    discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv.sort_values(by="clv", ascending=False).head(50)
    rfm_cltv_final = rfm.merge(cltv, on="Customer ID", how="left")
    rfm_cltv_final.sort_values(by="clv", ascending=False).head()
    return rfm_cltv_final


##############################################################
# 4. CLV Skoruna Göre Segmentlerin Oluşturulması
##############################################################
def scale_clv(df):
    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(df[["clv"]])
    df["scaled_clv"] = scaler.transform(df[["clv"]])
    return df

def segment_by_clv(df):
    df["clv_segment"] = pd.qcut(df["scaled_clv"], 4, labels=["bronze","silver", "gold", "premium"])
    premium_segment_ids = df[df["clv_segment"] == "premium"].index
    gold_segment_ids = df[df["clv_segment"] == "gold"].index
    silver_segment_ids = df[df["clv_segment"] == "silver"].index
    bronze_segment_ids = df[df["clv_segment"] == "bronze"].index
    return df, [premium_segment_ids,gold_segment_ids,silver_segment_ids,bronze_segment_ids]

##############################################################
# 5. Association Rule Learning
##############################################################
def create_invoice_product_df(dataframe,country):
    dataframe = dataframe[dataframe['Country'] == country]
    return dataframe.groupby(['Invoice', 'StockCode'])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0) 

    # We are filling NA values with zeros wich means that invoice does not contain that product
    # we are changing quantity value to (one ore zero). 
    # Because we are not interested in quantity of purchesed products. We just need it did purchesed or not

def create_rules(frequent_itemsets):
    dataframe = create_invoice_product_df(dataframe)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    # This function (apriori) creates support values which means purches rate of an item or item combinations
    # Also we gave a minimum support value to limit the return. 
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01) 
    # Again we determine a minimum treshold value for support metric
    return rules



def main():
    ##################### 1. READ DATA #############################
    #read the data with pandas method.
    df = read_data()
    print("1. step is done")
    
    ##################### 2. PREP DATA #############################
    #Preprocesses the data frame
    df = prep_data(df)
    print("2. step is done")
    
    ##################### 3. CREATE RFM DDF ########################
    # Creating a RFM DF having columns 'recency_cltv_p', 'tenure', 'frequency', 'monetary'
    rfm = rfm_df(df)
    print("3. step is done")
    
    ##################### 4. CREATE AND FIT BETA GEOMETRIC DISTRIBUTION  ##################
    #Create the BG/NBD model and fit
    bgf = fit_bgf(rfm)
    print("4. step is done")
    
    ##################### 5. PRED BETA  ############################
    # week/4 ay içinde en çok satın alma beklediğimiz n_cust müşteri kimdir?
    # Who are the 10 customers we expect the most to purchase in a 24 weeks?
    pred_bgf(bgf=bgf,rfm=rfm,week=24,n_cust=10)
    print("5. step is done")
    
    ##################### 6. TOP NTH CUSTOMER  MOST LIKELY TO PURCHASE ############################
    ##Who are the 10 customers we expect the most to purchase in a Input Week?
    rfm = exp_sales(bgf,rfm,week=24)
    print("6. step is done")
    
    ##################### 7. CALCULATE NUMBER OF SALES  ############################
    # What is the Expected Number of Sales of the Whole Company in 6 Months? 952.4548865072431
    expected_transaction(bgf,rfm,week=24)
    print("7. step is done")
    
    ##################### 8. EVALUATION OF BGF MODEL  ############################
    # Evaluation of Forecast Results
    eval_predictions(bgf)
    print("8. step is done")
    
    ########## 9. CREATE AND FIT GAMMA-GAMMA  ######################
    #Create the Gamma-Gamma model and fit
    # It is used for the estimation of the conditional expected average profit in a certain period of time.
    ggf = fit_ggf(rfm)
    print("9. step is done")
    
    ##################### 10. PRED GAMMA-GAMMA  #####################
    rfm = pred_ggf(ggf,rfm)
    print("10. step is done")
    
    ########## 11. PRINT RFM WITH EXPECTED AVERAGE PROFIT  ##########
    print(rfm.sort_values("expected_average_profit", ascending=False).head(20))
    print("11. step is done")
    
    ##################### 12. CREATE CLV DF  ########################
    # Set up a model that makes a 6-month CLTV prediction by combining our BG-NBD and Gamma-Gamma model. 
    # As a result, we will bring our customers who are most likely to shop in a 6-month projection
    rfm_cltv_final = calculate_clv(bgf,ggf,rfm,month=6)
    print("12. step is done")

    ############## 13. PRINT FINAL CLTV DF  #########################
    print(rfm_cltv_final.head())
    print(rfm_cltv_final.shape)
    print("13. step is done")

    ############## 14. SCALE CLTV AND CREATE SEGMENTS #########################
    rfm_cltv = scale_clv(rfm_cltv)
    rfm_cltv = segment_by_clv(rfm_cltv)

    ############## 15. CREATE INVOICE PRODUCT DF #########################
    df_pivot = create_invoice_product_df(df,"Germany")

    ############## 16. GET FREQUENT ITEMSETS #########################
    frequent_itemsets = get_frequent_itemset(df_pivot)


if __name__=='__main__':
    main()