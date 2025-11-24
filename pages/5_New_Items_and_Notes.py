import streamlit as st
note = st.text_area("Notes for future devs:", key="dev_notes", height=300)
st.caption(" Student Group 11203-Bibin Jose,Nicolas Vidal,Freddy Kuriakose")
if st.button("Save notes"):
    with open("developer_notes.txt", "a", encoding="utf-8") as f:
        f.write(note + "\n" + "-"*50 + "\n")
    st.success("Notes saved locally!")

