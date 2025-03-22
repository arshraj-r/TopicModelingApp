import streamlit as st
import pandas as pd
from bertopic_model import TopicModeling
from utils import save_to_excel
import datetime  
import plotly.express as px
import plotly.io as pio

def main():
    st.title("Topic Modeling Streamlit App")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Display the first few rows of the CSV
        st.write("Input Data head:")
        st.dataframe(df.head())

        # Display the column names and ask the user to select a column
        columns = df.columns.tolist()
        selected_column = st.selectbox("Select a column for topic modeling", columns)
        
        if selected_column:
            st.write(f"Performing topic modeling on the '{selected_column}' column.")
        
            # Perform topic modeling on the text column
            start_time=datetime.datetime.now()
            model= TopicModeling()
            topics, probabilities=model.fit(df[selected_column])
            
            # Add the results (topic and probabilities) to the DataFrame
            df['Topic'] = topics
            df['Probability'] = probabilities  # Get the max probability for each document
            
            #calculating time taken to run code
            end_time=datetime.datetime.now()
            time_taken = end_time - start_time
            time_taken_seconds = time_taken.total_seconds()
            
            # Display the DataFrame with the new columns
            st.write("Output Data with Topics and Probabilities, Time to Run :",time_taken_seconds)
            st.dataframe(df)

            # Save the results to an Excel file
            output_filename = save_to_excel(df)
            
            # Provide download link for the generated Excel file
            st.download_button(
                label="Download Output Excel File",
                data=open(output_filename, "rb").read(),
                file_name=output_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            fig1 = model.visualize_topics()  # Scatter plot of topics
            fig2 = model.visualize_barchart()  # Topic hierarchy (tree)
            fig3 = model.visualize_hierarchy()  # Documents scatter plot

            # Plot the visualizations using Streamlit
            st.subheader("Topic Visualization (Scatter Plot)")
            st.plotly_chart(fig1)  # Display the scatter plot of topics

            st.subheader("Topic Barchart")
            st.plotly_chart(fig2)  # Display the topic hierarchy

            st.subheader("Topic Hierarchy")
            st.plotly_chart(fig3)  

# Run the app
if __name__ == "__main__":
    main()