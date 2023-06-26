import clv

def main():
    ##################### READ DATA #############################
    df = clv.read_data()
    ##################### PREP DATA #############################
    df = clv.prep_data(df)
    ##################### CREATE RFM DDF ########################
    rfm = clv.rfm_df(df)
    ##################### CREATE AND FIT BETA  ##################
    bgf = clv.fit_bgf(rfm)
    ##################### PRED BETA  ############################
    clv.pred_bgf(bgf,rfm,24,10)
    ##################### PRED BETA  ############################
    rfm = clv.exp_sales(bgf,rfm,24)
    
    ##################### PRED BETA  ############################
    clv.expected_transaction(bgf,rfm,24)
    
    ##################### PRED BETA  ############################
    clv.eval_predictions(bgf)
    
    ##################### PRED BETA  ############################
    ggf = clv.fit_ggf(rfm)
    
    ##################### PRED BETA  ############################
    rfm = clv.pred_ggf(ggf,rfm)
    
    ##################### PRED BETA  ############################
    print(rfm.sort_values("expected_average_profit", ascending=False).head(20))
    
    ##################### PRED BETA  ############################
    rfm_cltv_final = clv.calculate_clv(bgf,ggf,rfm)

    ##################### PRED BETA  ############################
    print(rfm_cltv_final.head())
    print(rfm_cltv_final.shape)
    print(df.head())

if __name__=='__main__':
    main()