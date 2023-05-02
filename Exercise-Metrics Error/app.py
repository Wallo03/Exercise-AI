import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import mlem


def main():
    st.title("Exercise HeatMap")

    

    df = pd.read_csv('https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/Startup.csv')
    df_corr = df.corr()

    st.header('Date CSV of the StartUp')
    st.dataframe(df)

    st.header('The correlation of the StartUp')
    st.dataframe(df_corr)



    if st.button('Click me for the HeatMap'):
        
        st.subheader('HeatMap and Mlem')
        fig = plt.figure()
        sns.heatmap(df_corr, annot=True)
        st.pyplot(fig)

    #st.download_button(
        #label = "Download data as CSV",
        #data = df,
        #file_name = 'heatmap_df.csv',
        #mime = './Exercise-Regressione/app.py'
    #)

    rd_spend = st.number_input('Inserisci numero spese ricerca e sviluppo', 0, 10000, 5000)
    administration = st.number_input('Inserisci numero spese amministrazione', 0, 10000, 5000)
    marketing_spend = st.number_input('Inserisci numero spese marketing', 0, 10000, 5000)
    newmodel = mlem.api.load('model_.mlem')
    pred = newmodel.predict([[rd_spend,administration,marketing_spend]]) 
    
    st.write(pred[0])

    
        
    
        







if __name__ == "__main__":
    main()
