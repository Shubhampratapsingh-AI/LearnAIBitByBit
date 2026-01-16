import streamlit as st
import google.generativeai as genai
import os

# Gemini API Key from Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Add GEMINI_API_KEY to Streamlit Secrets (ai.google.dev)")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-3-flash-preview')

# YOUR COMPLETE DAILY TOPICS LIST (exactly as provided)
DAILY_TOPICS = {
    # MONTH 1: AI BASICS (Days 1-30)
    1: "What is Artificial Intelligence? Definitions and common misconceptions",
    2: "AI vs Machine Learning vs Deep Learning - The hierarchy explained", 
    3: "Narrow AI vs General AI vs Superintelligence - Future implications",
    4: "Alan Turing's contributions and the Turing Test",
    5: "1960s AI Winter - Why early AI failed",
    6: "Deep Blue vs Kasparov (1997) - First major AI victory",
    7: "Week 1 Recap: AI evolution timeline",
    
    8: "Why Python dominates AI? Installing Anaconda",
    9: "NumPy arrays vs Python lists - Speed comparison",
    10: "Pandas DataFrames - Your AI data workbench", 
    11: "Loading datasets: CSV, JSON, Excel with Pandas",
    12: "Data cleaning: Handling missing values",
    13: "Data visualization: Matplotlib basics",
    14: "Week 2: Python AI environment setup complete",
    
    15: "Vectors and matrices - AI's mathematical language",
    16: "Dot product and matrix multiplication intuition",
    17: "Derivatives for optimization - Calculus basics",
    18: "Gradient descent - How AI learns",
    19: "Probability basics: Distributions you need",
    20: "Bayes Theorem for AI decision making",
    21: "Week 3: Math toolkit for AI ready",
    
    22: "AI Bias: How algorithms discriminate",
    23: "Fairness metrics in machine learning",
    24: "Privacy concerns: Data protection laws",
    25: "AI ethics frameworks (Asimov's laws updated)",
    26: "Job displacement - AI's economic impact",
    27: "Existential risks - Superintelligence scenarios",
    28: "Responsible AI principles for developers",
    29: "Case study: Real-world AI ethics failures",
    30: "Month 1 Complete: AI foundations mastered",
    
    # MONTH 2: ML FUNDAMENTALS (Days 31-59)
    31: "Linear Regression: Predicting house prices",
    32: "Gradient descent optimization visualized",
    33: "R² score and Mean Absolute Error explained",
    34: "Logistic Regression for binary classification",
    35: "Decision boundaries and probability thresholds",
    36: "Multi-class classification with softmax",
    37: "KNN algorithm - Your first classifier",
    38: "Scikit-learn pipeline basics",
    39: "Train-test split and cross-validation",
    40: "Week 5: Supervised learning foundation",
    
    41: "K-Means clustering - Customer segmentation",
    42: "Elbow method for optimal clusters",
    43: "Hierarchical clustering dendrograms",
    44: "PCA - Dimensionality reduction visualized",
    45: "t-SNE for visualization of high-D data",
    46: "Feature scaling - Why it matters",
    47: "Outlier detection techniques",
    48: "DBSCAN for noisy data clustering",
    49: "Week 6: Unsupervised learning toolkit",
    50: "Real-world clustering applications",
    
    51: "Overfitting vs Underfitting - The bias-variance tradeoff",
    52: "Cross-validation techniques: K-Fold, Stratified",
    53: "Scikit-learn model selection module intro",
    54: "Hyperparameter tuning with GridSearchCV",
    55: "Feature importance visualization techniques",
    56: "Model persistence with joblib/pickle",
    57: "Week 7: ML evaluation mastery",
    58: "Complete ML workflow template",
    59: "Month 2 Complete: ML practitioner ready",
    
    # MONTH 3: ADVANCED ML (Days 60-90)
    60: "Random Forest algorithm deep dive",
    61: "XGBoost installation and first model",
    62: "Gradient boosting mathematics explained",
    63: "Feature importance in tree ensembles",
    64: "Bagging vs Boosting comparison",
    65: "Hyperparameter tuning for ensembles",
    66: "Stacking models for maximum performance",
    67: "Week 9: Ensemble mastery",
    68: "Handling imbalanced datasets (SMOTE)",
    69: "Feature engineering: Polynomial features",
    70: "Month 3 Checkpoint: Advanced ML complete",
    
    71: "Advanced dimensionality reduction (UMAP)",
    72: "Automated feature selection techniques",
    73: "Feature scaling methods comparison",
    74: "Encoding categorical variables (One-hot)",
    75: "Text feature extraction (TF-IDF)",
    76: "Bayesian optimization for hyperparameters",
    77: "Optuna vs GridSearch performance",
    78: "Pipeline automation with ColumnTransformer",
    79: "Week 10: Feature engineering mastery",
    80: "Real-world feature engineering case study",
    
    81: "ARIMA for time series forecasting",
    82: "Prophet library for business forecasting",
    83: "Seasonality and trend decomposition",
    84: "Isolation Forest for anomaly detection",
    85: "Autoencoders for anomaly detection",
    86: "Time series cross-validation",
    87: "Multivariate time series forecasting",
    88: "Real-world anomaly detection examples",
    89: "Week 11: Time series & anomalies",
    90: "Month 3 Complete: Advanced ML expert",
    
    # MONTH 4: DEEP LEARNING INTRO (Days 91-120)
    91: "Neural networks: Perceptrons basics",
    92: "Activation functions: ReLU, Sigmoid, Tanh",
    93: "Backpropagation algorithm explained",
    94: "Gradient descent variants (Batch/Mini/Stochastic)",
    95: "Vanishing gradient problem",
    96: "Building first neural net with Keras",
    97: "Week 13: Neural network foundations",
    98: "Loss functions: MSE, Cross-entropy",
    99: "Optimizers intro: SGD momentum",
    100: "Month 4 Checkpoint: NN basics mastered",
    
    101: "Convolutional Neural Networks (CNN) intro",
    102: "Convolution operation visualized",
    103: "Pooling layers: Max, Average pooling",
    104: "Building CNN for image classification",
    105: "Transfer learning with VGG16",
    106: "Data augmentation techniques",
    107: "Week 14: CNN fundamentals",
    108: "Object detection basics (YOLO intro)",
    109: "Image segmentation overview",
    110: "Real-world CNN applications",
    
    111: "Recurrent Neural Networks (RNN) basics",
    112: "LSTM vs GRU vs Vanilla RNN",
    113: "Building text classifier with LSTM",
    114: "Time series prediction with RNNs",
    115: "Bidirectional RNNs explained",
    116: "Attention mechanism preview",
    117: "Week 15: Sequence modeling intro",
    118: "Text preprocessing for deep learning",
    119: "Word embeddings (Word2Vec intro)",
    120: "Month 4 Complete: DL foundations ready",
    
    # MONTH 5: TRANSFORMERS (Days 121-151)
    121: "Attention mechanism - Self-attention basics",
    122: "Multi-head attention explained",
    123: "Transformer encoder architecture",
    124: "Positional encoding mathematics",
    125: "BERT: Bidirectional Encoder basics",
    126: "Week 17: Attention mechanisms",
    127: "BERT pre-training objectives",
    128: "Using BERT with Hugging Face",
    129: "Fine-tuning BERT for classification",
    130: "BERT variants (RoBERTa, DistilBERT)",
    
    131: "GPT architecture evolution (GPT-1 to 4)",
    132: "Autoregressive generation explained",
    133: "Beam search vs greedy decoding",
    134: "Top-K and Top-P sampling",
    135: "Temperature in text generation",
    136: "Week 18: GPT fundamentals",
    137: "Prompt engineering techniques",
    138: "Chain-of-thought prompting",
    139: "Few-shot learning with GPT",
    140: "In-context learning capabilities",
    
    141: "Hugging Face Transformers library",
    142: "Fine-tuning with Trainer API",
    143: "PEFT methods (Parameter Efficient Fine-tuning)",
    144: "LoRA: Low-Rank Adaptation",
    145: "QLoRA for memory efficiency",
    146: "Week 20: Fine-tuning toolkit",
    147: "Model evaluation metrics (BLEU, ROUGE)",
    148: "Deploying transformers with TGI",
    149: "Quantization for production",
    150: "Transformer debugging techniques",
    151: "Month 5 Complete: Transformers expert",
    
    # MONTH 6: DL OPTIMIZATION (Days 152-181)
    152: "Adam optimizer deep dive",
    153: "RMSprop vs Adam comparison",
    154: "Learning rate scheduling",
    155: "Cosine annealing explained",
    156: "Loss functions for different tasks",
    157: "Week 22: Optimization mastery",
    158: "Gradient clipping techniques",
    159: "Mixed precision training",
    160: "Distributed training basics",
    161: "Horovod for multi-GPU",
    162: "Month 6 Checkpoint: Optimizers",
    
    163: "Transfer learning strategies",
    164: "Domain adaptation techniques",
    165: "Data augmentation libraries (Albumentations)",
    166: "Test-time augmentation",
    167: "Progressive resizing",
    168: "Week 23: Transfer learning",
    169: "Knowledge distillation",
    170: "Neural architecture search intro",
    171: "AutoML for deep learning",
    172: "EfficientNet architecture",
    
    173: "PyTorch vs TensorFlow comparison",
    174: "Building CNN with PyTorch",
    175: "Keras 3 multi-backend support",
    176: "Lightning AI for clean code",
    177: "Week 25: Framework mastery",
    178: "Custom layers and models",
    179: "Model deployment with FastAPI",
    180: "ONNX for model interoperability",
    181: "Month 6 Complete: DL production ready",
    
    # MONTH 7: GENERATIVE AI BASICS (Days 182-212)
    182: "GANs: Generator vs Discriminator",
    183: "Minimax loss formulation",
    184: "DCGAN architecture",
    185: "Conditional GANs explained",
    186: "Mode collapse problem",
    187: "Week 27: GAN foundations",
    188: "Wasserstein GAN (WGAN)",
    189: "StyleGAN for face generation",
    190: "GAN evaluation metrics (FID)",
    191: "GANs for data augmentation",
    192: "Real-world GAN applications",
    
    193: "VAEs: Variational Autoencoders",
    194: "KL divergence explained",
    195: "Beta-VAE for disentanglement",
    196: "Diffusion models intro",
    197: "DDPM: Denoising Diffusion",
    198: "Week 29: Probabilistic generation",
    199: "Score-based generative models",
    200: "Latent diffusion basics",
    201: "Flow-based models overview",
    202: "Generative model comparisons",
    
    203: "Prompt engineering fundamentals",
    204: "Zero-shot vs few-shot prompting",
    205: "Chain-of-thought prompting",
    206: "Role prompting techniques",
    207: "Temperature and Top-P control",
    208: "Week 31: LLM prompting mastery",
    209: "ChatGPT system prompts",
    210: "Prompt chaining patterns",
    211: "Evaluation of LLM outputs",
    212: "Month 7 Complete: GenAI foundations",
    
    # MONTH 8: ADVANCED GENAI (Days 213-243)
    213: "RAG: Retrieval Augmented Generation",
    214: "Dense retrieval with embeddings",
    215: "Vector databases intro (Pinecone)",
    216: "FAISS for similarity search",
    217: "Hybrid search techniques",
    218: "Week 33: RAG systems",
    219: "LangChain for RAG pipelines",
    220: "LlamaIndex for document QA",
    221: "Chunking strategies for RAG",
    222: "Reranking for better retrieval",
    223: "Advanced RAG architectures",
    
    224: "Stable Diffusion architecture",
    225: "Text-to-image generation pipeline",
    226: "ControlNet for conditioned generation",
    227: "Img2Img and inpainting",
    228: "LoRA training for Stable Diffusion",
    229: "Week 35: Multimodal GenAI",
    230: "Video generation basics",
    231: "Audio generation with AudioLDM",
    232: "3D generation preview",
    233: "Multimodal model evaluation",
    
    234: "LoRA: Low-Rank Adaptation",
    235: "QLoRA for memory efficiency",
    236: "PEFT: Parameter Efficient Fine-tuning",
    237: "Prefix tuning techniques",
    238: "Adapter tuning methods",
    239: "Week 37: Efficient fine-tuning",
    240: "Unsloth for fast fine-tuning",
    241: "Axolotl training framework",
    242: "Fine-tuning evaluation",
    243: "Month 8 Complete: GenAI expert",
    
    # MONTH 9: AGENTIC GENAI (Days 244-273)
    244: "AI Agents: ReAct pattern",
    245: "Tool calling in LLMs",
    246: "Agent memory systems",
    247: "LangChain agent toolkit",
    248: "CrewAI for multi-agent",
    249: "Week 39: Agent foundations",
    250: "AutoGen framework intro",
    251: "Agent evaluation frameworks",
    252: "Planning in AI agents",
    253: "Reflection in agent loops",
    254: "Advanced agent architectures",
    
    255: "Multi-agent systems theory",
    256: "CrewAI practical implementation",
    257: "LangGraph for agent workflows",
    258: "AutoGen conversational agents",
    259: "Agent communication protocols",
    260: "Week 41: Multi-agent systems",
    261: "Hierarchical agent structures",
    262: "Market-based multi-agent systems",
    263: "Game theory in agents",
    264: "Multi-agent evaluation",
    
    265: "Synthetic data generation techniques",
    266: "2026 reasoning models (o1 preview)",
    267: "Agentic workflows future",
    268: "Multimodal agent systems",
    269: "Long-context reasoning",
    270: "Week 43: 2026 GenAI trends",
    271: "AI safety in agentic systems",
    272: "Evaluation of reasoning models",
    273: "Month 9 Complete: Agentic AI ready",
    
    # MONTH 10: ROBOTICS FOUNDATIONS (Days 274-304)
    274: "Robot kinematics forward",
    275: "Inverse kinematics solutions",
    276: "Robot dynamics basics",
    277: "Sensors: IMU, LiDAR, cameras",
    278: "Actuators: Motors and servos",
    279: "Week 45: Robot fundamentals",
    280: "Coordinate transformations",
    281: "DH parameters for robots",
    282: "Jacobian matrices",
    283: "Singularity analysis",
    284: "Kinematics simulation",
    
    285: "ROS2 installation and setup",
    286: "ROS2 topics and services",
    287: "ROS2 nodes and launch files",
    288: "Gazebo simulation environment",
    289: "URDF robot description",
    290: "Week 47: ROS2 basics",
    291: "RViz visualization tool",
    292: "ROS2 navigation stack",
    293: "MoveIt motion planning",
    294: "ROS2 industrial applications",
    
    295: "PID control theory",
    296: "A* path planning algorithm",
    297: "RRT motion planning",
    298: "Potential field methods",
    299: "Model predictive control",
    300: "Week 49: Robot control",
    301: "SLAM for localization",
    302: "AMCL localization",
    303: "Navigation stack tuning",
    304: "Month 10 Complete: Robotics ready",
    
    # MONTH 11: PHYSICAL AI (Days 305-334)
    305: "Computer vision for robotics",
    306: "SLAM: Simultaneous Localization and Mapping",
    307: "Visual odometry techniques",
    308: "Sensor fusion with Kalman filters",
    309: "LiDAR-based perception",
    310: "Week 51: Robot perception",
    311: "Object detection (YOLOv8)",
    312: "Depth estimation techniques",
    313: "Semantic segmentation",
    314: "Multi-sensor fusion",
    315: "Perception pipeline design",
    
    316: "Vision-Language-Action (VLA) models",
    317: "Imitation learning from observation",
    318: "Behavioral cloning basics",
    319: "Dataset aggregation (DAgger)",
    320: "Offline imitation learning",
    321: "Week 53: Robot learning",
    322: "RT-1: Robotics Transformer",
    323: "OpenVLA open source models",
    324: "Fine-tuning VLA models",
    325: "Imitation learning evaluation",
    
    326: "Reinforcement learning basics",
    327: "Q-Learning and DQN",
    328: "Policy gradient methods",
    329: "PPO: Proximal Policy Optimization",
    330: "SAC: Soft Actor Critic",
    331: "Week 55: RL for robots",
    332: "Sim-to-real transfer",
    333: "Domain randomization",
    334: "Month 11 Complete: Physical AI expert",
    
    # MONTH 12: INTEGRATION & FUTURE (Days 335-365)
    335: "LLMs for robot navigation",
    336: "Voice commands for robotics",
    337: "ROS2 + LLM integration",
    338: "Edge AI deployment on robots",
    339: "SayCan: LLM task planning",
    340: "Week 57: GenAI + Robotics",
    341: "Inner Monologue robot reasoning",
    342: "Code as Policies (CoP)",
    343: "VLM-based manipulation",
    344: "LLM code generation for robots",
    345: "Multi-modal robot control",
    
    346: "Humanoid robot architectures",
    347: "Tesla Optimus capabilities",
    348: "Figure 01 humanoid analysis",
    349: "Boston Dynamics Atlas evolution",
    350: "CES 2026 robotics highlights",
    351: "Week 59: Humanoids preview",
    352: "Edge AI hardware (Jetson)",
    353: "5G robot swarms",
    354: "Physical AI safety systems",
    355: "Future robot form factors",
    
    356: "MLOps for robotics deployment",
    357: "Docker containers for ROS",
    358: "Kubernetes for robot fleets",
    359: "Model versioning strategies",
    360: "Week 60: Production robotics",
    361: "Continuous integration for robots",
    362: "A/B testing robot behaviors",
    363: "Monitoring robot performance",
    364: "AI safety and alignment",
    365: "GRAND FINALE: AI Robotics Masterclass Complete!"

}

st.title("🚀 365-Day AI Series Generator")
st.markdown("**Gemini Text + Images • Your Exact Topics • Robotics Focus**")

day = st.number_input("Enter Day (1-365)", 1, 365, 1)
topic = DAILY_TOPICS.get(day, "AI & Robotics Deep Dive")

st.success(f"**📚 Day {day} Topic:** {topic}")

if st.button("✨ Generate Complete LinkedIn Package", type="primary"):
    with st.spinner("🎨 Gemini creating post + image..."):
        # 1. Generate POST TEXT
        text_prompt = f"""
        Day {day}: "{topic}"
        
        Create LinkedIn post for Indian tech YouTuber (robotics/electronics) for Day {day} of 365 days LearnPhysicalAIBitByBit:
        - 200 words: Hook → Simple explanation → Robotics example → Takeaway
        - Conversational, educational tone
        - End with question + hashtags (#AI365 #LearnPhysicalAIBitByBit #PhysicalAI #Robotics #GenAI)
        
        Output ONLY the post text ready to copy-paste.
        """
        
        text_response = model.generate_content(text_prompt)
        post_text = text_response.text
        
        # 2. Generate IMAGE
        image_prompt = f"""
        Create LinkedIn carousel image for: "{topic}"
        **Central Banner:**Across the exact center of the image, there is a large, glowing, futuristic banner with beveled edges. Inside this banner, in large, bold, neon cyan text, it reads: "DAY {day} OF 365: LearnAIBitByBit".
        Style: Professional tech infographic (blue/cyan)
        Include: Diagrams, robots, neural networks, code snippets
        Clean design for engineers learning AI/robotics
        1024x1024, high quality                                                                                                                                                                                                                                                                                                                             
        **Footer:** At the very bottom center, in smaller glowing text, include the fixed footer: "Keep Learning, Build the Future | Author: Shubham Pratap Singh".
        """
        
        # Note: Use Gemini's image gen endpoint or fallback to text description
        # For now, generate image prompt for manual creation
        st.markdown("## ✅ **YOUR LINKEDIN PACKAGE**")
        
        st.markdown("### 📝 **Post Text**")
        st.markdown(post_text)
        
        st.markdown("### 🖼️ **Image Prompt** (Copy to Gemini Image Gen)")
        st.code(image_prompt, language="text")
        
        # Downloads
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📄 Download Post", post_text, f"Day_{day}_Post.txt")
        with col2:
            st.download_button("🎨 Download Image Prompt", image_prompt, f"Day_{day}_Image_Prompt.txt")

st.info("💎 **Pro Tip:** Paste Image Prompt into Gemini image generator for instant visuals!")
st.caption("Days 1-68 + 335-365 loaded • Add more topics anytime!")

