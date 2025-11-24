import streamlit as st
st.set_page_config(page_title='New Items - Notes', layout='wide')
st.title('New Items / Ideas')
st.caption('Add new requirements or TODOs without touching the main flow.')
st.text_area('Notes for future devs:', key='dev_notes', height=300)
