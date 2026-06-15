import os
import json
import logging
from datetime import datetime
from curl_cffi import requests
from bs4 import BeautifulSoup
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

FUEL_PRICES_API_KEY = os.getenv("FUEL_PRICES_API_KEY")
if not FUEL_PRICES_API_KEY:
    logging.error("API_KEY not found in .env file")
    exit()

client = Groq(api_key=FUEL_PRICES_API_KEY)

def search_and_extract(query: str, prompt: str) -> dict:
    """Searching DuckDuckGo and extracting fuel prices using LLM"""
    try:
        url = "https://html.duckduckgo.com/html/"
        response = requests.get(url, params={"q": query}, impersonate="chrome", timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        snippets = [a.text for a in soup.find_all('a', class_='result__snippet')]
        
        if not snippets:
            return None
            
        context = "\n\n".join(snippets)
        
        response_llm = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Search Context:\n{context}"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response_llm.choices[0].message.content)
    except Exception as e:
        logging.error(f"Search/Extraction error: {e}")
        return None

def load_previous_prices():
    """Load last known fuel prices from file to fill in missing values if needed"""
    if os.path.exists("fuel_prices.json"):
        try:
            with open("fuel_prices.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def get_latest_prices():
    logging.info("Starting Zero-Touch Automated Fuel Price Extraction...")
    
    
    main_query = "اسعار البنزين 80 92 95 والسولار والغاز الطبيعي في مصر قرار وزارة البترول الجديد"
    main_prompt = """
    Extract current fuel prices in Egypt. Respond ONLY with valid JSON.
    Schema: {"solar": <number or null>, "benzine80": <number or null>, "benzine92": <number or null>, "benzine95": <number or null>, "naturalGas": <number or null>, "date": "<string or null>"}
    Rules: 
    - No hallucinations. Numbers only. Set to null if not found.
    - For "date": Extract the FULL date (day, month, year) from the text. Format: "DD Month YYYY" (e.g., "15 January 2025"). If only year is found, set to null.
    """
    
    result = search_and_extract(main_query, main_prompt)
    
    if not result:
        logging.error("Main search failed completely.")
        return load_previous_prices() 

    if result.get("naturalGas") is None:
        logging.warning("Natural Gas price is null. Triggering DEDICATED search for Natural Gas...")
        
        
        gas_query = "سعر متر الغاز الطبيعي للسيارات في محطات التموين مصر وزارة البترول"
        gas_prompt = """
        Extract ONLY the price of Natural Gas for cars in Egypt from the text.
        Respond ONLY with valid JSON.
        Schema: {"naturalGas": <number or null>, "date": "<string or null>"}
        Look for keywords like "متر الغاز", "غاز طبيعي", "سيارات".
        For "date": Extract the FULL date (day, month, year). Format: "DD Month YYYY". If only year is found, set to null.
        """
        
        gas_result = search_and_extract(gas_query, gas_prompt)
        
        if gas_result and gas_result.get("naturalGas") is not None:
            result["naturalGas"] = gas_result["naturalGas"]
            if gas_result.get("date"):
                result["date"] = gas_result["date"]

    missing_keys = [k for k in ["solar", "benzine80", "benzine92", "benzine95", "naturalGas"] if result.get(k) is None]
    
    if missing_keys:
        logging.warning(f"Search still missing: {missing_keys}. Loading last known good state from file...")
        previous_prices = load_previous_prices()
        
        if previous_prices:
            for key in missing_keys:
                result[key] = previous_prices.get(key)
            if not result.get("date"):
                result["date"] = previous_prices.get("date", "Last known valid state")
            logging.info("Successfully filled missing values from previous state.")
        else:
            logging.error("No previous state found. Some values will remain null.")

   
    current_date = result.get("date")
    if not current_date or (current_date.isdigit() and len(current_date) == 4):
       
        result["date"] = datetime.now().strftime("%d %B %Y")
        logging.info(f"Date not found or incomplete. Using current date: {result['date']}")

    return result

if __name__ == "__main__":
    prices = get_latest_prices()
    
    if prices:
        with open("fuel_prices.json", "w", encoding="utf-8") as f:
            json.dump(prices, f, ensure_ascii=False, indent=4)
        logging.info("Prices updated and saved successfully to fuel_prices.json")
        print(json.dumps(prices, ensure_ascii=False, indent=4))
    else:
        logging.error("Failed to retrieve any fuel prices.")
        
        
