import time

import requests
import pandas as pd

def fetch_steam_game_details(appid):
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        if data[str(appid)]["success"]:
            game_data = data[str(appid)]["data"]
            return {
                "steamId": appid,
                "name": game_data.get("name"),
                "release_date": game_data.get("release_date", {}).get("date"),
                "is_free": game_data.get("is_free"),
                "price": float(''.join(c for c in game_data.get("price_overview", {}).get("final_formatted") if c.isdigit() or c == '.'))
                            if not game_data.get("is_free") and game_data.get("price_overview", {}).get("final_formatted") else 0,
                "genres": ", ".join([genre["description"] for genre in game_data.get("genres", [])]),
                "metacritic_score": game_data.get("metacritic", {}).get("score"),
                "has_dlc": bool(game_data.get("dlc")),
                "has_demo": bool(game_data.get("demos")),
                "supported_platforms": [i for i, j in game_data.get("platforms").items() if j],
                "steam_achievements": dict(id=22, description="Steam Achievements") in game_data.get("categories"),
                "steam_trading_cards": dict(id=29, description="Steam Trading Cards") in game_data.get("categories"),
                "workshop_support": dict(id=30, description="Steam Workshop") in game_data.get("categories"),
                "achievements_total": game_data.get("achievements", {}).get("total", 0)
            }
        else:
            print(f"❌ AppID {appid} not found or data unavailable.")
            return None

    except Exception as e:
        print(f"⚠️ Error fetching data for AppID {appid}: {e}")
        return None


game_names = pd.read_csv("steam_id.csv")
games_details = list()
counter = 1
for i in game_names['steamId']:
    # Example usage
    if counter % 1000 == 0:
        pd.DataFrame(games_details).to_csv("extracted_data.csv", mode='a', header=False)
        games_details = list()

    if counter % 10 == 0:
        print(counter)
    appid = i  # Counter-Strike: Global Offensive
    game_info = fetch_steam_game_details(appid)

    if game_info:
        games_details.append(game_info)
        counter += 1
    time.sleep(1)

pd.DataFrame(games_details).to_csv("extracted_data.csv", mode='a', header=False)
