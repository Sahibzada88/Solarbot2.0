from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import pandas as pd
import os
from dotenv import load_dotenv
import re
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Load dataset
df = pd.read_csv("solar_data.csv")

# OpenAI Client (OpenRouter)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Request model
class UserQuery(BaseModel):
    question: str

# Extract location using regex
def extract_location(text: str) -> str:
    for location in df["Location"].unique():
        if re.search(rf"\b{location.lower()}\b", text.lower()):
            return location
    return None

# Extract number based on keywords
def extract_number_from_question(question: str, keyword: str) -> int:
    pattern = rf"{keyword}.*?(\d+)"
    match = re.search(pattern, question.lower())
    return int(match.group(1)) if match else None

# Core info fetcher
def get_solar_info_from_question(question: str) -> str:
    location = extract_location(question)
    bulbs = extract_number_from_question(question, "bulbs")
    fans = extract_number_from_question(question, "fans")
    fridges = extract_number_from_question(question, "fridges")
    acs = extract_number_from_question(question, "acs")
    pumps = extract_number_from_question(question, "motor pumps")

    filtered_df = df.copy()

    if location:
        filtered_df = filtered_df[filtered_df["Location"].str.lower() == location.lower()]
    if bulbs:
        filtered_df = filtered_df[filtered_df["Number_of_Bulbs"] >= bulbs]
    if fans:
        filtered_df = filtered_df[filtered_df["Number_of_Fans"] >= fans]
    if fridges:
        filtered_df = filtered_df[filtered_df["Number_of_Fridges"] >= fridges]
    if acs:
        filtered_df = filtered_df[filtered_df["Number_of_ACs"] >= acs]
    if pumps:
        filtered_df = filtered_df[filtered_df["Number_of_Motor_Pumps"] >= pumps]

    if not filtered_df.empty:
        row = filtered_df.iloc[0]
        return (
            f"In {row['Location'] if location else 'a recommended area'}, "
            f"avg sunlight: {row['Sunlight_Hours']} hours/day.\n"
            f"User appliances: bulbs={bulbs or 'N/A'}, fans={fans or 'N/A'}, "
            f"fridges={fridges or 'N/A'}, ACs={acs or 'N/A'}, pumps={pumps or 'N/A'}.\n"
            f"Recommended system: {row['Recommended_System']} for this setup."
        )
    else:
        return "Sorry, I couldn't find suitable solar system data based on the provided appliances."

@app.post("/ask")
async def ask_solar_bot(user_query: UserQuery):
    solar_info = get_solar_info_from_question(user_query.question)

    prompt = f"""
    Based on the following data from our solar dataset:
    {solar_info}

    The user asked: \"{user_query.question}\"

    Please respond with a clear, helpful solar system recommendation.
    """

    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"response": response.choices[0].message.content.replace("*", "")}
