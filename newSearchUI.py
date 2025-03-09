# Import required libraries
import streamlit as st
import os
import requests
import json

# Set up Streamlit app
st.title("Smart Search System")

# Define function to handle file upload
def handle_file_upload():
    uploaded_file = st.file_uploader("Upload TXT Files", accept_multiple_files=True)
    if uploaded_file is not None:
        with st.spinner("Uploading files..."):
            uploaded_files = [file for file in uploaded_file]
            for file in uploaded_files:
                file_path = os.path.join("./uploads", file.name)  # Place uploaded files in ./uploads directory
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            st.success("Files uploaded successfully!")
            return file_path

# Define function to execute search
def execute_search(query, file_paths):
    data = {"query": query, "documents": ",".join(file_paths)}
    headers = {"Content-Type": "application/json"}
    response = requests.post("http://your-backend-server:8000/search", data=json.dumps(data), headers=headers)
    results = response.json()
    return results

# Main app layout
st.sidebar.title("Navigation")
file_selection = st.sidebar.radio("Select an Option", ["Upload Files", "Search Documents"])

if file_selection == "Upload Files":
    file_path = handle_file_upload()
elif file_selection == "Search Documents":
    query_input = st.text_input("Enter your search query", "Your search query here")
    file_uploads = st.multiselect("Select uploaded files to search", options=[], key="files")
    if query_input and file_uploads:
        query = query_input
        file_paths = [os.path.join("./uploads", file) for file in file_uploads]
        search_results = execute_search(query, file_paths)

        # Display Search Results
        for result in search_results:
            st.write(f"Document: {result['filename']}")
            st.write(f"Relevance Score: {result['score']}")
            st.write("---")
            st.write(f"Content: {result['content']}")
            st.write(f"Sources: {', '.join(result['sources'])}")
            st.write("---")

# Persistence and static files handling
if not os.path.exists("./uploads"):
    os.makedirs("./uploads")

# Run Streamlit app
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.write("""
        <!DOCTYPE html>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            keyframe spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(359deg); }
            }
            #spinner {
                animation: spin 1s linear infinite;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 12px;
                height: 12px;
            }
        </style>
        <div id="spinner" style="display:none;">
            <div class="sk-circle1 sk-child"></div>
            <div class="sk-circle2 sk-child"></div>
            <div class="sk-circle3 sk-child"></div>
            <div class="sk-circle4 sk-child"></div>
        </div>
    """)
    st.markdown("<h1 style='text-align:center;'>Smart Search System</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='text-align:center;'>
        Upload text files to index and search through them using advanced querying techniques.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>⬅️ Go back to navigation</p>", unsafe_allow_html=True)
