from rfm import RFM
def main():
    rfm = RFM()
    rfm.data_prep()
    rfm.get_rfm_values()
    rfm.calculate_rfm_score()
    rfm.remove_outliers()
    rfm.plot_boxplot()
    return "Success!"

if __name__=='__main__':
    main()