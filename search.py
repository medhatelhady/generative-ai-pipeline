from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv

load_dotenv("config.env")

tavily_search = TavilySearch(
    max_results=5,
    topic="general",
    include_answer=False,
    include_raw_content=True,
    search_depth="advanced"
    #include_domains=["mayoclinic.org", "webmd.com", "nih.gov", "cdc.gov", "wikipedia.org", "healthline.com", "medicalnewstoday.com"]
)



def get_text(search_query):
    # # Create search query focusing on reputable medical sources
    # search_query = "electric vehicles"

    # Use Tavily search to find medical information
    search_results = tavily_search.invoke({"query": search_query})
    
    # Extract content from search results
    results_text = ""
    if search_results:
        for result in search_results.get("results", []):
            results_text += f"Source: {result.get('title', 'Unknown')}\n"
            results_text += f"Content: {result.get('content', '')}\n\n"

    return results_text

search_query = "electric vehicles"
text = get_text(search_query)

with open("search_result.txt", 'w') as f:
    f.write(text)