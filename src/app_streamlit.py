# src/app_streamlit.py
import streamlit as st
from advisor_agent import get_course_advice

# 🎓 --- App Title & Description ---
st.set_page_config(page_title="IITK Course Advisor", page_icon="🎓")

st.title("🎓 IITK AI Course Advisor")
st.markdown("""
Welcome to the **IIT Kanpur Course Advisor Chatbot**! 🤖  
This AI agent helps you make **smart course choices** using past grade trends, professor data, and student reviews.  

### 💡 What You Can Ask:
Try questions like:
- 🧠 *"Should I take MSE303?"*
- 📈 *"Is EE210 tough?"*
- 📊 *"Which AI/ML courses have good grading?"*
- 👨‍🏫 *"Professors who give high grades"*
- 🎯 *"Top 10 easiest scoring courses"*
- 🧾 *"Show grade distribution for ESC201"*

You’ll get insights like **average grades**, **professor grading styles**, and even **recommendations** about whether you should take a course — based on real IITK data.
""")

st.divider()

# 💬 --- Chat Input ---
query = st.text_input("💬 You:", placeholder="e.g., Average grade in AE201A or Should I take MSE303?")

# 🤖 --- Get AI Response ---
if query:
    with st.spinner("🔍 Thinking..."):
        reply = get_course_advice(query)
    st.markdown(f"**🎓 Advisor:** {reply}")
