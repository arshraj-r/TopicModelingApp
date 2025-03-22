import streamlit as st
import pandas as pd
from bertopic_model import BertopicModeling
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
        
        # Check if the input file has exactly two columns
        if df.shape[1] != 2:
            st.error("Please upload a CSV file with exactly two columns.")
            return
        
        # Display the first few rows of the CSV
        st.write("Input Data head:")
        st.dataframe(df.head())

        # Ensure the second column is the text column
        text_column = df.columns[0]
        st.write(f"Performing topic modeling on the '{text_column}' column.")
        
        # Perform topic modeling on the text column
        start_time=datetime.datetime.now()
        model= BertopicModeling()
        topics, probabilities=model.fit(df[text_column])
        
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
        if st.button("Save Topic Visualization (Scatter Plot)"):
            fig1.write_image("topic_visualization.png")
            st.success("Topic Visualization (Scatter Plot) saved as 'topic_visualization.png'")


        st.subheader("Topic Barchart")
        st.plotly_chart(fig2)  # Display the topic hierarchy
        # Save button for Topic Barchart
        if st.button("Save Topic Barchart"):
            fig2.write_image("topic_barchart.png")
            st.success("Topic Barchart saved as 'topic_barchart.png'")
            

        st.subheader("Topic Hierarchy")
        st.plotly_chart(fig3)  
        # Save button for Topic Hierarchy
        if st.button("Save Topic Hierarchy"):
            fig3.write_image("topic_hierarchy.png")
            st.success("Topic Hierarchy saved as 'topic_hierarchy.png'")

# Run the app
if __name__ == "__main__":
    main()