# 🎓 IIT Kanpur Course Advisor Chatbot

Welcome to the **IIT Kanpur Course Advisor Chatbot**! 🤖  
This AI assistant helps IITK students choose the right courses using:

✅ Past grade trends  
✅ Department patterns  
✅ Professor grading history  
✅ Student experience & reviews  

---

## 💡 What You Can Ask

You can chat naturally — try queries like:

| Query | What It Does |
|-------|--------------|
🧠 *"Should I take MSE303?"* | AI will analyze difficulty + grade trend + instructor style  
📈 *"Is EE210 tough?"* | Difficulty + historical grading pattern  
📊 *"Show grade distribution for ESC201"* | Plots grade distribution  
👨‍🏫 *"Professors who give high grades"* | Lists lenient graders  
🎯 *"Top 10 easiest scoring courses"* | Ranked suggestions  
📚 *"Best electives for ML"* | Course recommendations by domain  

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge)](https://profseek-tyb3d5a3dpnqvvfpyv4zs4.streamlit.app/)


## 🧠 Features

| Feature | Description |
|--------|-------------|
📊 Historical Grade Analytics | Uses IITK senate datasets  
🧑‍🏫 Professor Grade Tendencies | Identifies lenient vs strict graders  
⚖️ Difficulty Prediction | ML-based course toughness score  
✨ Natural Language Chat | Ask in English like a student  
🎨 Clean Web UI | Friendly interface for IITK students  

---

## 🚀 Tech Stack

| Layer | Tech |
|------|------|
Frontend | Next.js / Tailwind CSS  
AI Model | GPT-based Course Analysis + In-house Logic  
Data | IITK Senate Records + Student Review Dataset  
Backend | FastAPI / Node (depending on your setup)  

---

## 🔧 Installation (Dev Mode)

```bash
git clone https://github.com/YOUR_USERNAME/iitk-course-advisor.git
cd iitk-course-advisor
npm install
npm run dev

