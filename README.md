# Alpha Agent

An intelligent web research assistant that combines web crawling, search functionality, and AI-powered analysis using Anthropic's Claude API.

## Features

- Web crawling with intelligent content extraction
- Brave Search integration for finding relevant web pages
- AI-powered analysis using Claude 3
- Automatic query analysis and search decision making
- Conversation history tracking
- Token usage monitoring and cost calculation

## Prerequisites

- Python 3.11 or higher
- API keys for:
  - Anthropic Claude API
  - Brave Search API

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ghsaboias/alpha-agent.git
cd alpha-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```env
CLAUDE_API_KEY=your_claude_api_key
BRAVE_API_KEY=your_brave_api_key
```

## Usage

### Command Line Interface

Run a query using the command line interface:

```bash
python claude_api.py "your research query here"
```

The response will be saved to `claude_response.txt` and also printed to the console.

### Key Components

- `web_analyzer.py`: Handles web crawling and content analysis
- `claude_api.py`: Manages interactions with Claude API and orchestrates the research process
- `config.py`: Contains configuration settings and environment variables

### Configuration

You can modify various settings in `config.py`:

- Model selection (Claude 3 Haiku/Sonnet)
- Token limits
- Cost calculations
- Search result limits
- Logging levels

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your chosen license here]

## Acknowledgments

- [Anthropic](https://www.anthropic.com/) for the Claude API
- [Brave Search](https://brave.com/search/) for the search API
- [crawl4ai](https://github.com/crawl4ai/crawl4ai) for web crawling functionality
