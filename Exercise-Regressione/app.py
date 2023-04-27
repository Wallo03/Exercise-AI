import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    st.title("Exercise HeatMap")



    df = pd.read_csv('https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/Startup.csv')
    df_corr = df.corr()

    st.header('Date CSV of the StartUp')
    st.dataframe(df)

    st.header('The correlation of the StartUp')
    st.dataframe(df_corr)



    if st.button('Click me for the HeatMap'):
        
        st.subheader('HeatMap')
        fig = plt.figure()
        sns.heatmap(df_corr, annot=True)
        st.pyplot(fig)

    st.download_button(
        label = "Download data as CSV",
        data = df,
        file_name = 'heatmap_df.csv',
        mime = './Exercise-Regressione/app.py'
    )
        
    
        







if __name__ == "__main__":
    main()
