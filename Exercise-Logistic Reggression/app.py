import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():
    st.title("Exercise Logistic Reggression")
    
    df = pd.read_csv('iris.data', header=None)
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
    st.dataframe(df)
    y = df['class']
    X = df.drop(columns="class")
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state = 0
                                                    )
    model = LogisticRegression()
    model.fit(X_train, y_train)


    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        dfu = pd.read_csv(uploaded_file)
        st.dataframe(dfu)
        res = model.predict(dfu.to_numpy())
        #st.write(res)
        dfu['risultati'] = res
        st.dataframe(dfu)
    
        import io
        buffer = io.BytesIO()
    # download button 2 to download dataframe as xlsx
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Write each dataframe to a different worksheet.
            dfu.to_excel(writer, sheet_name='Sheet1', index=False)
            # Close the Pandas Excel writer and output the Excel file to the buffer
            writer.save()
            st.balloons()

            download2 = st.download_button(
                label="Download Excel",
                data=buffer,
                file_name='Risultati.xlsx',
                mime='application/vnd.ms-excel'
            )


    
if __name__ == "__main__":
    main()