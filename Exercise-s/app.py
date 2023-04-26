import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import streamlit as st

def main():
    st.title("Exercise")
    numpoint = st.slider('Inserisci Numero Punti', 1, 100, 50)
    coe = st.slider('Inserisci Coefficente Angolare', 1, 10,5)
    
    generate_random = np.random.RandomState(667)
    x = 10 * generate_random.rand(numpoint)
    y =  coe * x + np.random.randn(numpoint)


    X = x.reshape(-1, 1) # features
    y = y # target
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    y_pred = model.predict(X)

    fig = plt.figure(figsize = (10, 8))

    plt.scatter(x, y)
    plt.plot(x, y_pred,'-r')
    plt.title('Simple Linear Regression')
    st.pyplot(fig)

    st.write(f'')



if __name__ == "__main__":
    main()




