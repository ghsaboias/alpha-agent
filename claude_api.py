import anthropic, os, asyncio, json
from web_analyzer import SimpleCrawler, BraveSearch
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Tuple, Dict
import argparse

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
crawler = SimpleCrawler(os.getenv('CLAUDE_API_KEY'))
brave_search = BraveSearch()

conversation_history = {}
STORAGE_FILE = 'conversation_history.json'

def load_conversation_history():
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_conversation_history():
    with open(STORAGE_FILE, 'w') as f:
        json.dump(conversation_history, f)

conversation_history = load_conversation_history()

async def ask_claude(user_id, message):
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    time_info = f"Current date and time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"

    search_decision_prompt = f"""{time_info}

    Determine if a web search is needed to answer this query: "{message}"
    Your task is to decide whether current, real-time information from the internet is necessary to provide an accurate and complete answer.
    Respond with ONLY 'Yes' or 'No'.
    'Yes' if a search is needed (e.g., for current events, rapidly changing information, or specific facts you're unsure about).
    'No' if you can confidently answer based on your existing knowledge."""

    try:
        search_decision_response, _ = await _make_claude_call(
            "claude-3-opus-20240229",
            search_decision_prompt,
            "You are an AI assistant tasked with determining if a web search is needed."
        )
        
        search_decision = search_decision_response.content[0].text.strip().lower()
        search_needed = (search_decision == 'yes')
    except Exception as e:
        search_needed = True  # Default to searching if there's an error

    if search_needed:
        search_results = brave_search.search(message, count=2)
        crawl_results = await asyncio.gather(*[crawler.crawl_and_analyze(result['url'], message) for result in search_results])
        
        combined_info = "\n\n".join([
            f"URL: {result['url']}\n"
            f"Summary: {result['parsed_content'].get('summary', 'N/A')}\n"
            f"Key Points: {', '.join(result['parsed_content'].get('key_points', []))}\n"
            f"Time-sensitive Info: {result['parsed_content'].get('time_sensitive_info', 'N/A')}\n"
            f"Numerical Data: {json.dumps(result['parsed_content'].get('numerical_data', {}), indent=2)}\n"
            f"Sources: {', '.join(result['parsed_content'].get('sources', []))}"
            for result in crawl_results
        ])
        
        full_message = f"{time_info}\n\nRelevant information:\n\n{combined_info}\n\nUser query: {message}"
    else:
        full_message = f"{time_info}\n\n{message}"

    conversation_history[user_id].append({"role": "user", "content": full_message})
    save_conversation_history()

    try:
        response, token_usage = await _make_claude_call(
            "claude-3-5-sonnet-20240620",
            full_message,
            "You are an AI assistant providing quick and concise research. Use the provided information to answer the user's query accurately and concisely."
        )

        conversation_history[user_id].append({
            "role": "assistant", 
            "content": response.content[0].text,
            "token_usage": token_usage
        })
        save_conversation_history()
        return response.content[0].text
    except Exception as e:
        return f"Error processing request: {str(e)}"

async def _make_claude_call(model: str, prompt: str, system_message: str) -> Tuple[anthropic.types.Message, Dict]:
    response = client.beta.prompt_caching.messages.create(
        model=model,
        max_tokens=2000,
        system=[
            {"type": "text", "text": system_message},
            {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": "Please provide a comprehensive answer based on the information and instructions provided."}]
    )

    token_usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens
    }

    for attr in dir(response.usage):
        if attr.endswith('_tokens') and attr not in ['input_tokens', 'output_tokens']:
            token_usage[attr] = getattr(response.usage, attr)

    return response, token_usage

def clear_history(user_id):
    if user_id in conversation_history:
        conversation_history[user_id] = []
        save_conversation_history()
        return "Conversation history cleared."
    return "No conversation history found."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask Claude about a specific topic.")
    parser.add_argument("topic", help="The topic to ask Claude about")
    args = parser.parse_args()

    user_id = "test_user_123"
    test_message = args.topic
    
    async def main():
        response = await ask_claude(user_id, test_message)
        with open('claude_response.txt', 'w') as file:
            file.write(response)
        print(f"\nResponse from Claude: {response}")

    asyncio.run(main())
