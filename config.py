import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Discord settings
    DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
    GUILD_ID = int(os.getenv('GUILD_ID'))
    AUTHORIZED_USER_ID = int(os.getenv('AUTHORIZED_USER_ID'))

    # Claude API settings
    CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
    INITIAL_MODEL = "claude-3-haiku-20240307"
    FINAL_MODEL = "claude-3-5-sonnet-20240620"
    INITIAL_MAX_TOKENS = 1000
    FINAL_MAX_TOKENS = 4000

    # Conversation settings
    MAX_HISTORY = 10  # Maximum number of messages to keep in conversation history

    # Cost calculation (adjust these values based on actual pricing)
    COST_PER_1M_TOKENS = {
        'input': 0.01,  # $0.01 per 1M input tokens
        'output': 0.03  # $0.03 per 1M output tokens
    }

    @staticmethod
    def SYSTEM_PROMPT(current_date_time):
        return f"You are an AI assistant. The current date and time is {current_date_time}."

    # Web search settings
    MAX_SEARCH_RESULTS = 5  # This now represents the number of Google search results to crawl

    # Logging settings
    LOG_LEVEL = 'INFO'  # Can be DEBUG, INFO, WARNING, ERROR, CRITICAL