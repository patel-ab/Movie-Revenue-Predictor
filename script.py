import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re

# Helper function to convert Roman numerals to integers
def roman_to_int(s):
    roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total, prev_value = 0, 0
    for char in reversed(s):
        value = roman.get(char, 0)
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    return total

# Helper function to convert integers to Roman numerals
def int_to_roman(num):
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
    ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
    ]
    roman_numeral = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_numeral += syb[i]
            num -= val[i]
        i += 1
    return roman_numeral

# Helper function to generate patterns with numeral variants and optional year
def generate_patterns_with_numeral_variants(title, year=None):
    # Remove any quotation marks (single or double) from the title
    title = re.sub(r'[\'"]', '', title)

    # Use a set to ensure unique patterns
    patterns = {title}

    # Replace Roman numerals with integers
    roman_numerals = re.findall(r'\b[IVXLCDM]+\b', title)
    if roman_numerals:
        for numeral in roman_numerals:
            integer = roman_to_int(numeral)
            title_with_int = title.replace(numeral, str(integer))
            patterns.add(title_with_int)

    # Replace integers with Roman numerals
    integers = re.findall(r'\b\d+\b', title)
    if integers:
        for number in integers:
            roman_numeral = int_to_roman(int(number))
            title_with_roman = title.replace(number, roman_numeral)
            patterns.add(title_with_roman)

    # If a year is provided, add year-appended variants
    if year:
        patterns_with_year = {f"{pattern}_{year}" for pattern in patterns}
        patterns.update(patterns_with_year)

    return patterns

# Function to scrape Rotten Tomatoes ratings with multiple URL patterns
def get_rotten_tomatoes_rating(movie_name, year=None):
    try:
        if not isinstance(movie_name, (str, int)):
            return movie_name, 'NA', 'NA'

        movie_name = str(movie_name).strip()
        if not movie_name or movie_name.lower() == 'nan':
            return movie_name, 'NA', 'NA'

        # Generate title patterns with numeral variants and optional year
        base_patterns = generate_patterns_with_numeral_variants(movie_name, year)

        # Define possible URL patterns and use a set to ensure uniqueness
        url_patterns = set()
        for pattern in base_patterns:
            url_patterns.update([
                pattern.replace(" ", "_").lower(),
                pattern.replace(" ", "-").lower(),
                pattern.replace(":", "").replace(" ", "_").lower(),
                pattern.replace(":", "").replace(" ", "-").lower(),
                pattern.replace(".", "").replace(" ", "_").lower(),
                pattern.replace("-", "").replace(" ", "_").lower(),
            ])

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        for query in url_patterns:
            search_url = f"https://www.rottentomatoes.com/m/{query}"
            response = requests.get(search_url, headers=headers)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                audience_button = soup.find('rt-button', {'slot': 'audienceScore'})
                audience_score = audience_button.find('rt-text').get_text(strip=True) if audience_button else 'NA'

                critic_button = soup.find('rt-button', {'slot': 'criticsScore'})
                critic_score = critic_button.find('rt-text').get_text(strip=True) if critic_button else 'NA'

                return movie_name, audience_score, critic_score

        return movie_name, 'NA', 'NA'

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {movie_name}: {e}")
        return movie_name, 'NA', 'NA'

# Main function to fetch and save ratings
def fetch_and_save_ratings(input_file, output_file):
    df = pd.read_csv(input_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("id\tMovie Name\tAudience Score\tCritic Score\n")
        for _, row in df.iterrows():
            id = row['id']
            movie_name = row['original_title']
            release_date = row['release_date']

            # Extract year from release date
            year = str(release_date).split('-')[0] if pd.notna(release_date) else None

            # Fetch ratings with the extracted year
            movie_name, audience_score, critic_score = get_rotten_tomatoes_rating(movie_name, year)
            f.write(f"{id}\t{movie_name}\t{audience_score}\t{critic_score}\n")
            print(f"Fetched ratings for {movie_name}: Audience Score = {audience_score}, Critic Score = {critic_score}")
            time.sleep(1)  # Delay to avoid being blocked

# Usage
input_file = 'movie_dataset.csv'
output_file = 'rotten_tomatoes_movie_ratings_output.csv'
fetch_and_save_ratings(input_file, output_file)
