from groq import Groq # type:ignore
from json import load, dump
import datetime
from dotenv import dotenv_values # type:ignore

env_vars = dotenv_values(".env")

Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GropAPIKey = env_vars.get("GroqAPIKey")

client = Groq(api_key=GropAPIKey)

messages = []

System = f"""Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {Assistantname} which also has real-time up-to-date information from the internet.
*** Do not tell time until I ask, do not talk too much, just answer the question.***
*** Reply in only English, even if the question is in Hindi, reply in English.***
*** Do not provide notes in the output, just answer the question and never mention your training data. ***
"""

SystemChatBot = [
    {"role": "system", "content": System}
]

try:
    with open(r"Data/ChatLog.json","r") as f:
        messages = load(f)
except FileNotFoundError:
    with open(r"Data/ChatLog.json","w") as f:
        dump([],f)
        
def RealTimeInformation():
    current_date_time = datetime.datetime.now()
    day = current_date_time.strftime("%A")
    date = current_date_time.strftime("%d")
    month = current_date_time.strftime("%B")
    year = current_date_time.strftime("%Y")
    hour = current_date_time.strftime("%H")
    minute = current_date_time.strftime("%M")
    second = current_date_time.strftime("%S")
    
    data = f"Please use this real-time information if needed,\n"
    data += f"Day: {day}\nDate: {date}\nMonth: {month}\nYear: {year}\n"
    data += f"Time: {hour} hours : {minute} minutes : {second} seconds.\n"
    
    return data

def AnswerModifier(Answer):
    lines = Answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    modified_answer = '\n'.join(non_empty_lines)
    return modified_answer

def ChatBot(Query: str) -> str:
    """Send the user's query to the chatbot and return AI's response."""
    
    try:
        # Load previous chat
        try:
            with open(r"Data/ChatLog.json", "r") as f:
                chat_messages = load(f)
        except FileNotFoundError:
            chat_messages = []

        chat_messages.append({"role": "user", "content": Query})

        # Send request to AI
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=SystemChatBot + [{"role": "system", "content": RealTimeInformation()}] + chat_messages,
            max_tokens=300,
            temperature=0.6,
            top_p=1,
            stream=True,
            stop=None
        )

        Answer = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                Answer += chunk.choices[0].delta.content

        Answer = Answer.replace("</s>", "")
        chat_messages.append({"role": "assistant", "content": Answer})

        # Save chat history (append new messages)
        with open(r"Data/ChatLog.json", "w") as f:
            dump(chat_messages, f, indent=4)

        return AnswerModifier(Answer)

    except Exception as e:
        print(f"Error: {e}")
        # Do NOT clear JSON, just return error message
        return f"An error occurred: {e}"

    
if __name__ == "__main__":
    while True:
        user_input = input("Enter Your Question: ")
        print(ChatBot(user_input))