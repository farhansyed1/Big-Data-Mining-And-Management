from typing import Dict, List
from openai import AzureOpenAI
import sys
import math
import os
import re
import json
import pandas as pd

with open("api_key.json", "r") as f:
    data = json.load(f)
    azure_api_key = data["api_key"]

def fetch_restaurant_data(restaurant_name: str) -> Dict:
    """
    TODO: Fetch restaurant data from the csv file.
    Args:
        restaurant_name (str): Name of the restaurant.
    Returns:
        Dict: Disctionary containing restaurant data.
    Example:
        fetch_restaurant_data("Beyond Flavours")
        {
            "restaurant": "Beyond Flavours",
            "reviews": ["Great food!", "Excellent service", ... "Good ambiance"],
            "ratings": [5, 4, ..., 3],
            "metadata": ["1 Review , 2 Followers", "2 Reviews , 3 Followers", ... "4 Reviews , 5 Followers"]
        }
    """
    data = pd.read_csv('reviews.csv')
    filtered_data = data[data['Restaurant'].str.strip().str.lower() == restaurant_name.strip().lower()]

    result_dict = {
        "restaurant": restaurant_name,
        "reviews": filtered_data['Review'].tolist(),
        "ratings": filtered_data['Rating'].tolist(),
        "metadata": filtered_data['Metadata'].tolist()
    }

    return result_dict

def calculate_overall_rating(restaurant_name: str, ratings: List[int], followers: List[int]):
    """
    TODO: Calculate overall rating of the restaurant.
    For the given restaurant, the overall rating is calculated as follows:
    1. Calculate the weight for each rating based on the number of followers. The weight is calculated as:
        weight = math.log(followers + 1)
    2. Calculate the overall rating using the weighted average formula:
        overall_rating = sum(rating * weight) / sum(weight)
    Args:
        restaurant_name (str): Name of the restaurant.
        ratings (List[int]): List of ratings.
        followers (List[int]): List of followers.
    Returns:
        float: Overall rating of the restaurant. The result should be rounded to 3 decimal places. For example: if the result is 3.14159, it should return 3.142.
    """
    weights = [math.log(f + 1) for f in followers]  
    weighted_sum = sum(r * w for r, w in zip(ratings, weights))
    return round(weighted_sum / sum(weights), 3)

class IntentionRecognitionAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_version="2023-05-15",
            azure_endpoint="https://hkust.azure-api.net",
            api_key=azure_api_key,
        )
        
    def generate_response(self, user_input: str) -> str:
        """
        Args:
            user_input (str): User input.
        Returns:
            json string: Intention of the user.
        Example: 
            generate_response("What is the rating of Beyond Flavours?")
            {"intention": "rating", "args": ["Beyond Flavours"]}
            generate_response("Beyond Flavours and Karachi Cafe, which one is better?")
            {"intention": "compare", "args": ["Beyond Flavours", "Karachi Cafe"]}
            generate_response("What are people saying about Beyond Flavours?")
            {"intention": "review", "args": ["Beyond Flavours"]}
        """
        # TODO: write the prompt for recognizing the intention of the user.
        prompt = f"""
        Analyze this restaurant-related query and classify its intention:
        - "rating": if asking about a restaurant's rating/score
        - "compare": if comparing two restaurants
        - "review": if asking about reviews/opinions
        
        Also extract all restaurant names mentioned.
        
        Respond STRICTLY in this JSON format:
        {{
            "intention": "rating|compare|review",
            "args": ["restaurant_name1", "restaurant_name2"] 
        }}
        
        Examples:
        Input: "What's the rating for SKYHY?"
        Output: {{"intention": "rating", "args": ["SKYHY"]}}
        
        Input: "Which is better, SKYHY or Olive Garden?"
        Output: {{"intention": "compare", "args": ["SKYHY", "Olive Garden"]}}
        
        Input: "What do people say about SKYHY?"
        Output: {{"intention": "review", "args": ["SKYHY"]}}
        
        Now analyze this query:
        "{user_input}"
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.0,
            max_tokens=1000
        )
        response_text = response.choices[0].message.content

        return json.loads(response_text)

class ReviewSummaryAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_version="2023-05-15",
            azure_endpoint="https://hkust.azure-api.net",
            api_key=azure_api_key,
        )
        
    def generate_response(self, reviews: List[str]):
        """
        Args:
            reviews (List[str]): List of reviews.
        Returns:
            str: Summary of the reviews.
        Example:
            generate_response(["Great food!", "Excellent service", "Good ambiance"])
            returns "The restaurant has great food, excellent service, and good ambiance."
        """
        # TODO: write the prompt for summarizing the reviews.
        prompt = f"""
        You are a restaurant review summarizer. Please analyze the following reviews and 
        create a concise and neutral summary that captures the main sentiments expressed.
        Focus on common themes and overall impressions.
        
        Reviews:
        {reviews}
        
        Provide your summary in 1-2 sentences, starting with "The reviews suggest that..." 
        or similar phrasing. Be objective and don't invent details not present in the reviews.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.0,
            max_tokens=1000,
        )
        response_text = response.choices[0].message.content
        # Since the response is a summary, we can assume it's already in the desired format and doesn't need further parsing.
        return response_text.strip()
    
class FollowersParserAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_version="2023-05-15",
            azure_endpoint="https://hkust.azure-api.net",
            api_key=azure_api_key,
        )
        
    def generate_response(self, metadata: List[str]):
        """
        Args:
            metadata (List[str]): List of metadata.
        Returns:
            List[int]: List of followers.
        Example:
            generate_response(["1 Review , 2 Followers", "2 Reviews , 3 Followers", "4 Reviews , 5 Followers"])
            returns [2, 3, 5]
        """
        # TODO: write the prompt for parsing the metadata.
        prompt = f"""
        Extract just the follower counts from each of these metadata strings and return them as a JSON list of integers.
        The strings follow the pattern "X Review(s) , Y Follower(s)".
        Only return the numbers after "Followers" in each string.
        
        Example Input: ["1 Review , 2 Followers", "2 Reviews , 3 Followers", "4 Reviews , 5 Followers"]
        Example Output: [2, 3, 5]
        
        Now process this input:
        {metadata}
        
        Return ONLY the JSON array of follower counts, nothing else.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.0,
            max_tokens=1000,
        )
        response_text = response.choices[0].message.content
        # TODO: parse the response text to extract the followers
        try:
            # If output is JSON 
            return json.loads(response_text)
        except Exception as e:
            # Use regex if fail
            followers = []
            for item in metadata:
                match = re.search(r'(\d+)\s*Follower(s)?', item, re.IGNORECASE)
                if match:
                    followers.append(int(match.group(1)))
            return followers

def main(user_query: str)-> str:
    # Initialize the agents
    intention_recognition_agent = IntentionRecognitionAgent()
    review_summary_agent = ReviewSummaryAgent()
    follower_parser = FollowersParserAgent()
    
    # Step 1: Recognize the intention of the user
    intention = intention_recognition_agent.generate_response(user_query)
    if intention["intention"].lower() == "rating":
        # Step 2: Fetch restaurant data
        restaurant_name = intention["args"][0]
        restaurant_data = fetch_restaurant_data(restaurant_name)
        
        # Step 3: Calculate overall rating
        ratings = restaurant_data["ratings"]
        followers = follower_parser.generate_response(restaurant_data["metadata"])
        overall_rating = calculate_overall_rating(restaurant_name, ratings, followers)
        
        # Step 4: Generate response
        response = f"The overall rating of {restaurant_name} is {overall_rating}."
        return response
    elif intention["intention"].lower() == "compare":
        # Step 2: Fetch restaurant data
        restaurant_name1 = intention["args"][0]
        restaurant_name2 = intention["args"][1]
        restaurant_data1 = fetch_restaurant_data(restaurant_name1)
        restaurant_data2 = fetch_restaurant_data(restaurant_name2)
        
        # Step 3: Calculate overall rating
        ratings1 = restaurant_data1["ratings"]
        followers1 = follower_parser.generate_response(restaurant_data1["metadata"])
        overall_rating1 = calculate_overall_rating(restaurant_name1, ratings1, followers1)
        ratings2 = restaurant_data2["ratings"]
        followers2 = follower_parser.generate_response(restaurant_data2["metadata"])
        overall_rating2 = calculate_overall_rating(restaurant_name2, ratings2, followers2)
        # Step 4: Generate response
        if overall_rating1 > overall_rating2:
            response = f"{restaurant_name1} has a higher rating than {restaurant_name2}, so you should go for {restaurant_name1}."
        elif overall_rating1 < overall_rating2:
            response = f"{restaurant_name2} has a higher rating than {restaurant_name1}, so you should go for {restaurant_name2}."
        else:
            response = f"{restaurant_name1} and {restaurant_name2} have the same rating, so you can choose either."
        return response
    elif intention["intention"].lower() == "review":
        # Step 2: Fetch restaurant data
        restaurant_name = intention["args"][0]
        restaurant_data = fetch_restaurant_data(restaurant_name)
        
        # Step 3: Generate response
        reviews = restaurant_data["reviews"]
        response = review_summary_agent.generate_response(reviews)
        return response
    else:
        raise ValueError(f"Invalid intention recognized. Intention: {intention['intention']}, Expected: rating, compare, review.")

    
if __name__ == "__main__":
    while True:
        user_query = input("USER(Enter your query or EXIT to exit): ")
        if user_query.strip().lower() == "exit":
            break
        response = main(user_query)
        print(f"SYSTEM: {response}")
