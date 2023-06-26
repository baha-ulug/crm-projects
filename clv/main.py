import clv

def main():
    ##################### 1. READ DATA #############################
    #read the data with pandas method.
    df = clv.read_data()
    print("1. step is done")
    
    ##################### 2. PREP DATA #############################
    #Preprocesses the data frame
    df = clv.prep_data(df)
    print("2. step is done")
    
    ##################### 3. CREATE RFM DDF ########################
    # Creating a RFM DF having columns 'recency_cltv_p', 'tenure', 'frequency', 'monetary'
    rfm = clv.rfm_df(df)
    print("3. step is done")
    
    ##################### 4. CREATE AND FIT BETA  ##################
    #Create the BG/NBD model and fit
    bgf = clv.fit_bgf(rfm)
    print("4. step is done")
    
    ##################### 5. PRED BETA  ############################
    # week/4 ay içinde en çok satın alma beklediğimiz n_cust müşteri kimdir?
    # Who are the 10 customers we expect the most to purchase in a 24 weeks?
    clv.pred_bgf(bgf=bgf,rfm=rfm,week=24,n_cust=10)
    print("5. step is done")
    
    ##################### 6. TOP NTH CUSTOMER  MOST LIKELT TO PURCHASE ############################
    ##Who are the 10 customers we expect the most to purchase in a Input Week?
    rfm = clv.exp_sales(bgf,rfm,week=24)
    print("6. step is done")
    
    ##################### 7. CALCULATE NUMBER OF SALES  ############################
    # What is the Expected Number of Sales of the Whole Company in 6 Months? 952.4548865072431
    clv.expected_transaction(bgf,rfm,week=24)
    print("7. step is done")
    
    ##################### 8. EVALUATION OF BGF MODEL  ############################
    # Evaluation of Forecast Results
    clv.eval_predictions(bgf)
    print("8. step is done")
    
    ########## 9. CREATE AND FIT GAMMA-GAMMA  ######################
    #Create the Gamma-Gamma model and fit
    # It is used for the estimation of the conditional expected average profit in a certain period of time.
    ggf = clv.fit_ggf(rfm)
    print("9. step is done")
    
    ##################### 10. PRED GAMMA-GAMMA  #####################
    rfm = clv.pred_ggf(ggf,rfm)
    print("10. step is done")
    
    ########## 11. PRINT RFM WITH EXPECTED AVERAGE PROFIT  ##########
    print(rfm.sort_values("expected_average_profit", ascending=False).head(20))
    print("11. step is done")
    
    ##################### 12. CREATE CLV DF  ########################
    # Set up a model that makes a 6-month CLTV prediction by combining our BG-NBD and Gamma-Gamma model. 
    # As a result, we will bring our customers who are most likely to shop in a 6-month projection
    rfm_cltv_final = clv.calculate_clv(bgf,ggf,rfm,month=6)
    print("12. step is done")

    ############## 13. PRINT FINAL CLTV DF  #########################
    print(rfm_cltv_final.head())
    print(rfm_cltv_final.shape)
    print("13. step is done")

if __name__=='__main__':
    main()