import os, requests, json, anthropic
from crawl4ai import WebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
import time
import random
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import traceback

class BraveSearch:
    def __init__(self):
        self.api_key = os.getenv("BRAVE_API_KEY")
        self.search_url = "https://api.search.brave.com/res/v1/web/search"
    
    def search(self, query: str, count: int = 5):
        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        params = {'q': query, 'count': count, 'key': self.api_key}
        response = requests.get(self.search_url, headers=headers, params=params)
        if response.status_code == 200:
            results = response.json().get('web', {}).get('results', [])
            return [{"url": result['url'], "title": result['title']} for result in results]
        return []

class QueryAnalyzer:
    def __init__(self, client):
        self.client = client

    def analyze_query(self, query: str) -> dict:
        prompt = f"""
        Analyze the following query: "{query}"
        Provide a JSON output with the following structure:
        {{
            "intent": "The primary intent of the query",
            "keywords": ["List", "of", "important", "keywords"],
            "domain": "The general domain or topic of the query",
            "complexity": "A score from 1-5 indicating query complexity"
        }}
        """
        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return json.loads(response.content[0].text)

class SimpleCrawler:
    def __init__(self, api_key: str):
        self.crawler = WebCrawler()
        self.crawler.warmup()
        self.client = anthropic.Anthropic(api_key=api_key)
        self.query_analyzer = QueryAnalyzer(self.client)

    def generate_extraction_instruction(self, query: str, query_analysis: dict) -> str:
        prompt = f"""
        Generate an extraction instruction for a web crawler based on:
        Query: "{query}"
        Intent: {query_analysis['intent']}
        Domain: {query_analysis['domain']}
        Complexity: {query_analysis['complexity']}

        Create a detailed instruction focusing on extracting:
        1. Relevant facts and data
        2. Time-sensitive information if applicable
        3. Different perspectives or interpretations, if relevant
        4. Numerical data and statistics when appropriate
        5. Sources and citations for key information

        Provide the instruction in plain text.
        """
        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text.strip()

    def determine_crawl_depth(self, query_analysis: dict) -> int:
        base_depth = 2
        if query_analysis['complexity'] > 3:
            base_depth += 1
        if query_analysis['domain'] in ['science', 'technology', 'finance', 'health']:
            base_depth += 1
        return min(base_depth, 5)  # Cap at 5 to avoid excessive crawling

    def is_query_answered(self, crawled_content: dict, query: str) -> bool:
        prompt = f"""
        Given the following crawled content:
        {json.dumps(crawled_content, indent=2)}

        Determine if this content sufficiently answers the query: "{query}"
        Respond with ONLY 'Yes' if the content fully answers the query, or 'No' if more information is needed.
        """
        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text.strip().lower() == 'yes'

    async def crawl_and_analyze(self, url: str, query: str):
        try:
            print(f"Starting crawl for URL: {url}")
            query_analysis = self.query_analyzer.analyze_query(query)
            print(f"Query analysis for '{query}': {query_analysis}")

            extraction_instruction = self.generate_extraction_instruction(query, query_analysis)
            print(f"Extraction instruction: {extraction_instruction}")

            print(f"Initiating crawler.run for {url}")
            result = self.crawler.run(
                url=url,
                word_count_threshold=300,  # Increased to capture more content in fewer blocks
                extraction_strategy=LLMExtractionStrategy(
                    provider="anthropic/claude-3-haiku-20240307",
                    api_token=self.client.api_key,
                    schema={
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "key_points": {"type": "array", "items": {"type": "string"}},
                            "time_sensitive_info": {"type": "string"},
                            "numerical_data": {"type": "object"},
                            "sources": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["summary", "key_points"]
                    },
                    instruction=extraction_instruction
                ),
                bypass_cache=True,
                max_depth=1  # Limit to one depth initially
            )
            print(f"Crawler.run completed for {url}")
            
            print(f"Extracted content from {url}: {len(result.extracted_content)} blocks")
            print(f"First 500 characters of extracted content: {str(result.extracted_content)[:500]}")

            parsed_content = self._parse_extracted_content(result.extracted_content)
            print(f"Parsed content: {parsed_content}")

            is_answered = self.is_query_answered(parsed_content, query)
            print(f"Is query answered: {is_answered}")
            
            crawl_data = {
                "url": url,
                "query": query,
                "parsed_content": parsed_content,
                "is_answered": is_answered
            }
            self._save_crawl_data(crawl_data)
            print(f"Crawl data saved for {url}")
            return crawl_data

        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")
            print(f"Exception type: {type(e)}")
            print(f"Exception traceback: {traceback.format_exc()}")
            error_data = {"url": url, "query": query, "extracted_content": {}, "parsed_content": {}, "error": str(e)}
            self._save_crawl_data(error_data)
            print(f"Error data saved for {url}")
            return error_data

    def _process_block(self, block, query):
        parsed_content = self._parse_extracted_content(block)
        is_answered = self.is_query_answered(parsed_content, query)
        print(f"Claude response for query '{query}': {parsed_content}")
        print(f"Is query answered: {is_answered}")
        return parsed_content, is_answered

    def _parse_extracted_content(self, extracted_content):
        if not extracted_content:
            return {"error": "No content extracted"}
        
        if isinstance(extracted_content, dict):
            if "data" in extracted_content:
                return self._parse_data_field(extracted_content["data"])
            elif "summary" in extracted_content and "key_points" in extracted_content:
                return extracted_content
        elif isinstance(extracted_content, str):
            return self._parse_string_content(extracted_content)
        return {"error": "Unrecognized content format"}

    def _parse_data_field(self, data):
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
                if isinstance(parsed_data, list):
                    return self._combine_list_items(parsed_data)
                return parsed_data
            except json.JSONDecodeError:
                return self._parse_string_content(data)
        elif isinstance(data, list):
            return self._combine_list_items(data)
        return {"error": "Unexpected data format"}

    def _combine_list_items(self, items):
        combined = {"summary": "", "key_points": []}
        for item in items:
            if isinstance(item, dict):
                combined["summary"] += item.get("summary", "") + " "
                combined["key_points"].extend(item.get("key_points", []))
        combined["summary"] = combined["summary"].strip()
        return combined

    def _parse_string_content(self, content):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return self._combine_list_items(parsed)
            return parsed
        except json.JSONDecodeError:
            # If it's not valid JSON, try to extract meaningful content
            lines = content.split('\n')
            summary = next((line for line in lines if line.strip().startswith("summary")), "")
            key_points = [line.strip() for line in lines if line.strip().startswith("-")]
            return {"summary": summary, "key_points": key_points}

    def _cached_llm_call(self, content: str, query: str):
        max_retries = 5
        base_delay = 1
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    system=f"Provide information related to {query}. Output in JSON format only.",
                    messages=[
                        {"role": "user", "content": "Analyze the content and provide the requested information in JSON format only."},
                        {"role": "user", "content": content, "cache_control": {"type": "ephemeral"}}
                    ]
                )
                
                raw_response = response.content[0].text
                return self._parse_llm_response(raw_response), self._get_token_usage(response)
            except anthropic.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                delay = (2 ** attempt) * base_delay + random.uniform(0, 1)
                print(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        return {}, {}

    def _parse_llm_response(self, raw_response):
        # Try to handle different response types
        if isinstance(raw_response, str):
            # Check if it's a JSON object or list
            if raw_response.startswith("{") or raw_response.startswith("["):
                try:
                    parsed_response = json.loads(raw_response)
                    # Handle parsed JSON (list, dict, etc.)
                    if isinstance(parsed_response, dict):
                        return self._handle_dict_response(parsed_response)
                    elif isinstance(parsed_response, list):
                        return self._handle_list_response(parsed_response)
                except json.JSONDecodeError:
                    # Invalid JSON, try to handle as text
                    return {"error": "Failed to parse JSON response", "raw_response": raw_response}
            else:
                # Handle other string responses (e.g., "Here is: {}")
                return self._handle_string_response(raw_response)
        return {"error": "Unrecognized response format"}

    def _handle_dict_response(self, response_dict):
        # Check if the dictionary contains expected keys
        if "summary" in response_dict and "key_points" in response_dict:
            return {
                "summary": response_dict.get("summary", ""),
                "key_points": response_dict.get("key_points", []),
                "error": False
            }
        # Handle other cases
        return {"error": "Unexpected dictionary format", "content": response_dict}

    def _handle_list_response(self, response_list):
        # Handle empty or invalid lists
        if not response_list:
            return {"error": "Empty list response"}
        return {"data": response_list}

    def _handle_string_response(self, response_str):
        # Remove placeholders like "{}" or "[]"
        cleaned_response = response_str.replace("{}", "").replace("[]", "").strip()
        if cleaned_response:
            return {"data": cleaned_response}
        return {"error": "Empty response content"}

    def _get_token_usage(self, response):
        usage = response.usage
        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens
        }

    def _save_crawl_data(self, data):
        # Create a directory to store crawl results if it doesn't exist
        os.makedirs("crawl_results", exist_ok=True)

        # Generate a filename based on the URL
        filename = f"crawl_results/crawl_result_{data['url'].replace('://', '_').replace('/', '_')}.json"

        # Save the data to a JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4, default=str)

        print(f"Crawl result saved to {filename}")

if __name__ == "__main__":
    api_key = os.getenv("ANTHROPIC_API_KEY")
    crawler = SimpleCrawler(api_key)
    brave_search = BraveSearch()
    
    search_query = "gold price"
    search_results = brave_search.search(query=search_query, count=2)
    
    async def main():
        tasks = [crawler.crawl_and_analyze(result['url'], search_query) for result in search_results]
        await asyncio.gather(*tasks)

    asyncio.run(main())
