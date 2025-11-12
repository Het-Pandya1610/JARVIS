import datetime
from googlesearch import search  # type:ignore
from groq import Groq, RateLimitError  # type:ignore
from json import load, dump
from dotenv import dotenv_values  # type:ignore
import time

# 🔹 Load environment variables
env_vars = dotenv_values(".env")
Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
GroqAPIKey = env_vars.get("GroqAPIKey")

client = Groq(api_key=GroqAPIKey)

System = """"""  # Your system instructions here

# 🔹 Load chat log once
try:
    with open(r"Data/ChatLog.json", "r") as f:
        messages = load(f)
except:
    messages = []
    with open(r"Data/ChatLog.json", "w") as f:
        dump(messages, f, indent=2)

# 🔹 Fast Google Search (just URLs, limited)
def GoogleSearch(query, max_results=3):
    try:
        results = list(search(query, num_results=max_results))
    except Exception as e:
        results = []
    Answer = f"Search results for '{query}':\n[start]\n"
    for url in results:
        Answer += f"URL: {url}\n"
    Answer += "[end]"
    return Answer

# 🔹 Clean answer
def AnswerModifier(answer):
    return "\n".join([line for line in answer.split("\n") if line.strip()])

# 🔹 Initial system conversation
SystemChatBot = [
    {"role": "system", "content": System},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello, how can I help you?"}
]

# 🔹 Real-time information
def Information():
    now = datetime.datetime.now()
    return (
        f"Day: {now.strftime('%A')}, "
        f"Date: {now.strftime('%d-%B-%Y')}, "
        f"Time: {now.strftime('%H:%M:%S')}"
    )

# 🔹 Safe completion to handle rate limits
def safe_completion(**kwargs):
    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            msg = str(e)
            # Extract wait time from the error message
            if "Please try again in" in msg:
                wait_time = float(msg.split("Please try again in ")[1].split("s")[0])
            else:
                wait_time = 15  # default wait
            print(f"Rate limit reached. Waiting {wait_time + 1:.1f}s before retrying...")
            time.sleep(wait_time + 1)

# 🔹 Main function
def RealTimeSearchEngine(prompt):
    global SystemChatBot, messages

    # Append user query
    messages.append({"role": "user", "content": prompt})

    # Add Google search results only for the current prompt
    search_result = GoogleSearch(prompt)
    SystemChatBot.append({"role": "system", "content": search_result})

    # Use only the last 5 messages to reduce token usage
    recent_messages = messages[-5:]

    # Generate completion safely
    completion = safe_completion(
        model="llama-3.3-70b-versatile",
        messages=SystemChatBot + [{"role": "system", "content": Information()}] + recent_messages,
        max_tokens=512,
        temperature=0.6,
        top_p=1,
        stream=False
    )

    Answer = completion.choices[0].message.content.strip().replace("</s>", "")
    messages.append({"role": "assistant", "content": Answer})

    # Save log
    with open(r"Data/ChatLog.json", "w") as f:
        dump(messages, f, indent=2)

    # Remove last system Google search message to avoid duplicating
    SystemChatBot.pop()

    return AnswerModifier(Answer)

# 🔹 Run in loop
if __name__ == "__main__":
    while True:
        prompt = input("Enter Your Query: ")
        print(RealTimeSearchEngine(prompt))
