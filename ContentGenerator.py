import streamlit as st
import google.generativeai as genai
import json

# Paste your Gemini API key here
GEMINI_API_KEY = "AIzaSyAJM4cfZ1zE4lmM1Ai5_X4d5mvoAAouZPI"  # Replace with your key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Embedded 365-day topics (condensed from your series outline)
TOPICS = {
    "1-7": "AI Basics: What is AI? History, types (narrow vs general), real-world impact.",
    "8-14": "Python for AI: NumPy, Pandas basics, data handling.",
    "15-21": "Math essentials: Linear algebra, calculus, probability.",
    "22-30": "AI Ethics, bias, future effects.",
    "31-59": "ML Fundamentals: Supervised/unsupervised learning, scikit-learn.",
    "60-90": "Advanced ML: Ensembles, feature engineering.",
    "91-120": "Deep Learning Intro: Neural nets, CNNs, RNNs.",
    "121-151": "Transformers: Attention, BERT, GPT basics.",
    "152-181": "DL Optimization: Optimizers, transfer learning.",
    "182-212": "Generative AI Basics: GANs, VAEs, prompt engineering.",
    "213-243": "Advanced GenAI: RAG, multimodal, fine-tuning.",
    "244-273": "Agentic GenAI: Agents, multi-agent systems.",
    "274-304": "Robotics Foundations: Kinematics, ROS, control.",
    "305-334": "Physical AI: Perception, SLAM, RL for robots.",
    "335-365": "Integration: GenAI in robotics, trends, projects."
}

MONTH_GROUPS = {
    1: "1-30", 2: "31-59", 3: "60-90", 4: "91-120", 5: "121-151", 6: "152-181",
    7: "182-212", 8: "213-243", 9: "244-273", 10: "274-304", 11: "305-334", 12: "335-365"
}

st.title("🚀 365-Day AI Series LinkedIn Post Generator")
st.markdown("Enter day number (1-365) for instant post draft powered by Gemini!")

day = st.number_input("Day Number", min_value=1, max_value=365, value=1)

if st.button("Generate Post"):
    month = ((day - 1) // 30) + 1
    topic_group = MONTH_GROUPS.get(month, "1-30")
    topic = TOPICS.get(topic_group, "AI Fundamentals")
    
    prompt = f"""
    Create a engaging LinkedIn post for Day {day} of a 365-day AI series by an Indian tech YouTuber focused on electronics/robotics.
    Topic: {topic}
    Style: Educational, bite-sized (200-300 words), hook + explanation + robotics/genAI example + CTA.
    End with 3-5 hashtags like #AI365 #PhysicalAI. Make it perfect for robotics enthusiasts learning from basics to advanced.
    Output ONLY the post text.
    """
    
    response = model.generate_content(prompt)
    post = response.text
    
    st.markdown("### 📝 Generated Post")
    st.markdown(post)
    st.markdown("---")
    st.markdown("**Copy-paste ready! Customize if needed.**")
    
    # Download option
    st.download_button("Download Post", post, f"AI_Series_Day_{day}.txt")

st.info("💡 Pro Tip: Deploy free on Streamlit Cloud for mobile access. Update TOPICS dict with more details from your outline!")
