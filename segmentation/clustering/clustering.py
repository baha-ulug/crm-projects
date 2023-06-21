import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def gen_df():
    # Örnek veri oluşturma
    np.random.seed(42)
    n_samples = 1000

    # Müşteri ID'leri
    customer_ids = np.arange(1000, 5000)

    # Satış verileri
    sales_mean = np.random.randint(low=100, high=1000, size=len(customer_ids))
    sales_std = sales_mean * 0.2
    sales = np.random.normal(loc=sales_mean, scale=sales_std)

    # Satın alma frekansı
    purchase_frequency_mean = np.random.randint(low=1, high=10, size=len(customer_ids))
    purchase_frequency_std = purchase_frequency_mean * 0.15
    purchase_frequency = np.random.normal(loc=purchase_frequency_mean, scale=purchase_frequency_std)

    # Ortalama sepet değeri
    average_basket_value_mean = np.random.uniform(low=10, high=100, size=len(customer_ids))
    average_basket_value_std = average_basket_value_mean * 0.3
    average_basket_value = np.random.normal(loc=average_basket_value_mean, scale=average_basket_value_std)

    # İade oranı
    return_rate = np.random.beta(2, 5, size=len(customer_ids))

    # Veri çerçevesini oluşturma
    df = pd.DataFrame({
        'sales_income': sales,
        'purchase_frequency': purchase_frequency,
        'average_basket_value': average_basket_value,
        'return_rate': return_rate
    })
    return df

def create_clusters(df):
    X = df[df.columns]

    # Verileri ölçeklendirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means modelini oluşturma ve uygulama
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)

    # Her bir müşteriye ait tahmini segmentleri alın
    segments = kmeans.predict(X_scaled)

    # Segmentlerin DataFrame'e eklenmesi
    df['segment'] = segments

    return df

def observe_kpi(df):
    # Segmentlere ait istatistiksel bilgilerin hesaplanması
    segment_stats = df.groupby('segment')[df.columns].mean()

    # KPI'ları hesaplama ve yazdırma
    customer_count = df['segment'].value_counts()
    print("Müşteri Sayısı:")
    print(customer_count)

    print("\nSatış Geliri:")
    print(segment_stats['sales_income'])

    print("\nSatın Alma Frekansı:")
    print(segment_stats['purchase_frequency'])

    print("\nOrtalama Sepet Değeri:")
    print(segment_stats['average_basket_value'])

    print("\nİade Oranı:")
    print(segment_stats['return_rate'])

if __name__=='__main__':
    df = gen_df()
    df_with_clusters = create_clusters(df)
    observe_kpi(df)