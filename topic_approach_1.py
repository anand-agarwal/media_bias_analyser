import ollama
import pandas as pd
from tqdm import tqdm
import json

# Load dataset
df = pd.read_csv("/Users/anandagarwal/Media Bias Analyser/processed_media_data_economictimes.csv")
new_df = df[:30]
new_df.to_csv("test.csv", index=False)

# Initialize topics list
topics_list = ["Women's Empowerment and Social Justice in India's Budget 2024"]

# Sample article for context
sample_article = """
The Union Budget 2024–25 epitomises the government’s commitment to balancing high capital expenditures with robust welfare spending. This budget is designed to empower every section of society, driving socio-economic growth and development across the nation. It will empower every section of society. It reflects the government's dedication to ensuring that all Indians, irrespective of religion, caste, gender, and age, make substantial progress in realising their life goals and aspirations. The government has placed a strong emphasis on inclusive growth, focusing on four major segments of society: Garib (poor), Mahilayen (women), Yuva (youth), and Annadata (farmers).

Despite global economic uncertainties, including elevated asset prices, political instability, and shipping disruptions, India’s economic growth remains a beacon of stability. India's inflation remains low and stable, moving towards the 4% target, with core inflation (non-food, non-fuel) currently at 3.1%. Proactive government steps to ensure adequate market supplies of perishable goods have contributed significantly to this stability.
The budget marks a significant stride in promoting socio-economic development, with a strong focus on women's empowerment and social justice. The ministry of women and child development's budget has increased by 3%, from ₹25,449 crore in FY 23–24 to ₹26,092 crore in FY 24–25. Meanwhile, the department of social justice and empowerment has seen a substantial 32% increase, from ₹9,853 crore in FY 23–24 to ₹13,000 crore in FY 24–25. The budget for the department of empowerment of persons with disabilities remains unchanged at ₹1,225 crore.

The government has allocated over ₹3 lakh crore for schemes benefiting women and girls, aiming to foster women-led development. This historic allocation emphasises the government's commitment to empowering women and enhancing their role in the workforce. Key initiatives include the establishment of working women hostels in collaboration with industry partners and the creation of creches to facilitate higher participation of women in the workforce. These hostels will provide safe and convenient living arrangements for working women, while creches will support working mothers by offering reliable childcare services.

Furthermore, the government plans to partner with industries to organise women-specific skilling programmes and promote market access for women-led Self-Help Group (SHG) enterprises. These initiatives are designed to equip women with the skills needed to thrive in the workforce and support their entrepreneurial ventures, thereby driving economic growth and development.
To achieve social justice comprehensively, the government will adopt a saturation approach, ensuring that all eligible individuals are covered by various programmes, including those focused on education and health. This approach aims to empower marginalised communities by improving their capabilities and providing them with the tools needed to succeed.
"""

# Combine text and title
new_df["combined"] = new_df["Text"] + " " + new_df["Title"]

def classify_article(article_text, topics_list):
    prompt = f"""
    You are an expert in natural language processing and manual text classification.
    
    I am trying to perform topic modelling on news articles related to the Indian Budget 2024 and cluster them based on topics. Analyze the provided article based on content, keywords, and context and provide the relevant topic.
    
    First check the existing topics list: {topics_list}. If the article matches one of the topics, return that topic. If the article does not match any topic, generate a new topic and append it to the list.
    
    For context, here is a sample article with its assigned topic:
    
    Sample Article:
     {sample_article}
    
    Assigned Topic: "Women's Empowerment and Social Justice in India's Budget 2024"
    
    Now classify the following article:
    {article_text}
    
    Return ONLY the following as a JSON object, without any explanations or additional text:
    {{
        "article_topic": "<determined_topic>",
        "updated_topics_list": [<updated_list_of_topics>]
    }}
    """
    
    response = ollama.chat(model="llama3.1", messages=[{"role": "user", "content": prompt}])
    response_text = response["message"]["content"].strip()
    print(response)
    
    try:
        response_json = json.loads(response_text)# Safe parsing
        return response_json["article_topic"], response_json["updated_topics_list"]
    except json.JSONDecodeError:
        print(f"Error parsing response: {response_text}")  # Debugging
        return "Unknown", topics_list  # Default values in case of failure

# Classify each article and update topics list
article_topics = []
for i in tqdm(new_df["combined"], desc="Classifying articles"):
    article_topic, topics_list = classify_article(i, topics_list)
    article_topics.append(article_topic)

# Add topics to dataframe
print(article_topics)
new_df["topic"] = article_topics

# Save the updated dataframe
new_df.to_csv("classified_articles.csv", index=False)
