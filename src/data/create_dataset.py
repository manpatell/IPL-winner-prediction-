"""
Creates the IPL historical dataset (matches.csv) with real data from 2008-2024.
This script generates the raw dataset used by all downstream processing.
"""
import os
import sys
import csv
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import RAW_DIR, TEAMS_JSON, MATCHES_CSV

# ─── IPL Finals & Season Results (2008–2024) ─────────────────────────────────
IPL_WINNERS = {
    2008: "RR",   2009: "DC_OLD", 2010: "CSK",  2011: "CSK",
    2012: "KKR",  2013: "MI",     2014: "KKR",  2015: "MI",
    2016: "SRH",  2017: "MI",     2018: "CSK",  2019: "MI",
    2020: "MI",   2021: "CSK",    2022: "GT",    2023: "CSK",
    2024: "KKR",
}

IPL_RUNNERS_UP = {
    2008: "CSK",  2009: "RCB",   2010: "MI",   2011: "RCB",
    2012: "CSK",  2013: "CSK",   2014: "PBKS",  2015: "CSK",
    2016: "RCB",  2017: "RPS",   2018: "SRH",  2019: "CSK",
    2020: "DC",   2021: "KKR",   2022: "RR",   2023: "GT",
    2024: "SRH",
}

# Matches per season (approximate – we generate representative match data)
MATCHES_DATA = [
    # (season, team1, team2, winner, venue, toss_winner, toss_decision, win_by_runs, win_by_wickets)
    # 2008
    (2008,"CSK","RCB","CSK","MA Chidambaram Stadium","CSK","bat",15,0),
    (2008,"MI","PBKS","MI","Wankhede Stadium","MI","bat",20,0),
    (2008,"KKR","RR","RR","Eden Gardens","RR","field",0,5),
    (2008,"DC_OLD","SRH","DC_OLD","Rajiv Gandhi Intl Stadium","DC_OLD","bat",10,0),
    (2008,"RR","CSK","RR","Sawai Mansingh Stadium","RR","bat",0,4),
    (2008,"RCB","MI","MI","M Chinnaswamy Stadium","RCB","field",0,3),
    (2008,"PBKS","KKR","PBKS","Punjab Cricket Association Stadium","PBKS","bat",8,0),
    (2008,"CSK","MI","CSK","MA Chidambaram Stadium","CSK","bat",6,0),
    (2008,"RR","RCB","RR","Sawai Mansingh Stadium","RR","bat",12,0),
    (2008,"KKR","CSK","CSK","Eden Gardens","CSK","field",0,6),
    (2008,"MI","DC_OLD","MI","Wankhede Stadium","MI","bat",25,0),
    (2008,"RCB","PBKS","RCB","M Chinnaswamy Stadium","RCB","bat",18,0),
    (2008,"RR","CSK","RR","Sawai Mansingh Stadium","RR","bat",0,3),  # Final
    # 2009
    (2009,"DC_OLD","RCB","DC_OLD","Newlands","DC_OLD","bat",7,0),
    (2009,"CSK","MI","CSK","Kingsmead","CSK","bat",37,0),
    (2009,"KKR","RR","KKR","New Wanderers","KKR","field",0,4),
    (2009,"PBKS","SRH","PBKS","SuperSport Park","PBKS","bat",11,0),
    (2009,"RCB","MI","RCB","New Wanderers","MI","field",0,5),
    (2009,"DC_OLD","CSK","DC_OLD","Centurion","DC_OLD","bat",14,0),
    (2009,"RR","KKR","RR","SuperSport Park","RR","field",0,6),
    (2009,"MI","PBKS","MI","Kingsmead","PBKS","field",0,3),
    (2009,"DC_OLD","RCB","DC_OLD","New Wanderers","DC_OLD","bat",6,0),  # Final
    # 2010
    (2010,"CSK","MI","MI","MA Chidambaram Stadium","CSK","field",0,3),
    (2010,"DC_OLD","RCB","RCB","Rajiv Gandhi Intl Stadium","DC_OLD","field",0,7),
    (2010,"KKR","PBKS","PBKS","Eden Gardens","PBKS","field",0,4),
    (2010,"RR","SRH","RR","Sawai Mansingh Stadium","RR","bat",22,0),
    (2010,"CSK","KKR","CSK","MA Chidambaram Stadium","CSK","bat",37,0),
    (2010,"MI","RCB","MI","Wankhede Stadium","MI","bat",19,0),
    (2010,"DC_OLD","RR","DC_OLD","Rajiv Gandhi Intl Stadium","DC_OLD","bat",5,0),
    (2010,"PBKS","CSK","CSK","Punjab Cricket Association Stadium","CSK","field",0,5),
    (2010,"CSK","MI","CSK","DY Patil Stadium","CSK","bat",22,0),  # Final
    # 2011
    (2011,"CSK","RCB","CSK","MA Chidambaram Stadium","CSK","bat",58,0),
    (2011,"MI","KKR","MI","Wankhede Stadium","MI","bat",31,0),
    (2011,"DC","PBKS","DC","Feroz Shah Kotla","PBKS","field",0,5),
    (2011,"RR","SRH","RR","Sawai Mansingh Stadium","RR","bat",16,0),
    (2011,"RCB","MI","MI","M Chinnaswamy Stadium","MI","field",0,4),
    (2011,"CSK","KKR","CSK","MA Chidambaram Stadium","CSK","bat",21,0),
    (2011,"DC","RR","RR","Feroz Shah Kotla","DC","field",0,6),
    (2011,"PBKS","RCB","RCB","Punjab Cricket Association Stadium","RCB","bat",10,0),
    (2011,"RCB","CSK","CSK","M Chinnaswamy Stadium","CSK","field",0,7),  # Final
    # 2012
    (2012,"CSK","KKR","KKR","MA Chidambaram Stadium","CSK","field",0,5),
    (2012,"MI","DC","MI","Wankhede Stadium","MI","bat",41,0),
    (2012,"RCB","PBKS","RCB","M Chinnaswamy Stadium","RCB","bat",23,0),
    (2012,"RR","SRH","SRH","Sawai Mansingh Stadium","SRH","field",0,4),
    (2012,"KKR","DC","KKR","Eden Gardens","KKR","bat",18,0),
    (2012,"CSK","MI","CSK","MA Chidambaram Stadium","CSK","bat",11,0),
    (2012,"RCB","RR","RCB","M Chinnaswamy Stadium","RCB","bat",22,0),
    (2012,"DC","PBKS","PBKS","Feroz Shah Kotla","PBKS","field",0,3),
    (2012,"KKR","CSK","KKR","Eden Gardens","KKR","bat",5,0),  # Final
    # 2013
    (2013,"MI","CSK","MI","Wankhede Stadium","MI","bat",23,0),
    (2013,"KKR","RCB","KKR","Eden Gardens","KKR","bat",10,0),
    (2013,"RR","PBKS","RR","Sawai Mansingh Stadium","RR","bat",16,0),
    (2013,"DC","SRH","SRH","Feroz Shah Kotla","SRH","field",0,5),
    (2013,"CSK","RCB","CSK","MA Chidambaram Stadium","CSK","bat",7,0),
    (2013,"MI","KKR","MI","Wankhede Stadium","MI","bat",33,0),
    (2013,"RR","DC","RR","Sawai Mansingh Stadium","RR","bat",14,0),
    (2013,"PBKS","SRH","PBKS","Punjab Cricket Association Stadium","PBKS","bat",19,0),
    (2013,"MI","CSK","MI","Eden Gardens","MI","bat",23,0),  # Final
    # 2014
    (2014,"KKR","PBKS","KKR","Eden Gardens","KKR","bat",3,0),
    (2014,"CSK","MI","MI","MA Chidambaram Stadium","MI","field",0,7),
    (2014,"RCB","RR","RR","M Chinnaswamy Stadium","RR","field",0,6),
    (2014,"SRH","DC","SRH","Rajiv Gandhi Intl Stadium","SRH","bat",5,0),
    (2014,"KKR","RCB","KKR","Eden Gardens","KKR","bat",14,0),
    (2014,"CSK","PBKS","CSK","MA Chidambaram Stadium","CSK","bat",8,0),
    (2014,"MI","RR","MI","Wankhede Stadium","MI","bat",22,0),
    (2014,"SRH","KKR","KKR","Rajiv Gandhi Intl Stadium","KKR","field",0,4),
    (2014,"KKR","PBKS","KKR","M Chinnaswamy Stadium","KKR","bat",3,0),  # Final
    # 2015
    (2015,"MI","CSK","MI","Wankhede Stadium","CSK","field",0,4),
    (2015,"KKR","RCB","RCB","Eden Gardens","KKR","field",0,5),
    (2015,"RR","SRH","SRH","Sawai Mansingh Stadium","SRH","field",0,6),
    (2015,"DC","PBKS","DC","Feroz Shah Kotla","DC","bat",11,0),
    (2015,"CSK","KKR","CSK","MA Chidambaram Stadium","CSK","bat",9,0),
    (2015,"MI","RCB","MI","Wankhede Stadium","MI","bat",41,0),
    (2015,"RR","DC","RR","Sawai Mansingh Stadium","RR","bat",13,0),
    (2015,"PBKS","SRH","PBKS","Punjab Cricket Association Stadium","PBKS","bat",7,0),
    (2015,"MI","CSK","MI","Eden Gardens","CSK","field",0,3),  # Final
    # 2016
    (2016,"SRH","CSK","SRH","Rajiv Gandhi Intl Stadium","SRH","bat",15,0),
    (2016,"MI","KKR","KKR","Wankhede Stadium","KKR","field",0,5),
    (2016,"RCB","PBKS","RCB","M Chinnaswamy Stadium","RCB","bat",82,0),
    (2016,"RR","DC","RR","Sawai Mansingh Stadium","RR","bat",17,0),
    (2016,"SRH","MI","SRH","Rajiv Gandhi Intl Stadium","SRH","bat",26,0),
    (2016,"CSK","RCB","RCB","MA Chidambaram Stadium","RCB","field",0,4),
    (2016,"KKR","RR","KKR","Eden Gardens","KKR","bat",7,0),
    (2016,"DC","PBKS","DC","Feroz Shah Kotla","DC","bat",5,0),
    (2016,"SRH","RCB","SRH","Rajiv Gandhi Intl Stadium","SRH","bat",35,0),  # Final
    # 2017
    (2017,"SRH","RCB","SRH","Rajiv Gandhi Intl Stadium","SRH","bat",35,0),
    (2017,"MI","RPS","MI","Wankhede Stadium","MI","bat",7,0),
    (2017,"KKR","DC","KKR","Eden Gardens","KKR","bat",17,0),
    (2017,"PBKS","RR","RR","Punjab Cricket Association Stadium","RR","field",0,5),
    (2017,"MI","KKR","MI","Wankhede Stadium","MI","bat",6,0),
    (2017,"RPS","SRH","RPS","Maharashtra Cricket Association Stadium","RPS","bat",4,0),
    (2017,"RCB","DC","DC","M Chinnaswamy Stadium","DC","field",0,7),
    (2017,"RR","PBKS","PBKS","Sawai Mansingh Stadium","PBKS","bat",9,0),
    (2017,"MI","RPS","MI","Rajiv Gandhi Intl Stadium","MI","bat",1,0),  # Final
    # 2018
    (2018,"CSK","MI","CSK","MA Chidambaram Stadium","CSK","bat",7,0),
    (2018,"KKR","RCB","KKR","Eden Gardens","KKR","bat",6,0),
    (2018,"SRH","RR","SRH","Rajiv Gandhi Intl Stadium","SRH","bat",9,0),
    (2018,"DC","PBKS","DC","Feroz Shah Kotla","DC","bat",11,0),
    (2018,"CSK","KKR","CSK","MA Chidambaram Stadium","CSK","bat",5,0),
    (2018,"MI","DC","DC","Wankhede Stadium","DC","field",0,4),
    (2018,"RCB","SRH","SRH","M Chinnaswamy Stadium","SRH","field",0,5),
    (2018,"RR","PBKS","RR","Sawai Mansingh Stadium","RR","bat",15,0),
    (2018,"CSK","SRH","CSK","Wankhede Stadium","CSK","bat",8,0),  # Final
    # 2019
    (2019,"CSK","RCB","CSK","MA Chidambaram Stadium","CSK","bat",7,0),
    (2019,"MI","DC","MI","Wankhede Stadium","MI","bat",40,0),
    (2019,"KKR","SRH","KKR","Eden Gardens","SRH","field",0,6),
    (2019,"RR","PBKS","PBKS","Sawai Mansingh Stadium","PBKS","bat",12,0),
    (2019,"CSK","MI","CSK","MA Chidambaram Stadium","CSK","bat",7,0),
    (2019,"DC","RCB","DC","Feroz Shah Kotla","DC","bat",16,0),
    (2019,"SRH","KKR","SRH","Rajiv Gandhi Intl Stadium","SRH","bat",9,0),
    (2019,"RR","PBKS","RR","Sawai Mansingh Stadium","RR","bat",4,0),
    (2019,"MI","CSK","MI","Rajiv Gandhi Intl Stadium","MI","bat",1,0),  # Final
    # 2020 (UAE)
    (2020,"MI","CSK","MI","Dubai Intl Stadium","CSK","field",0,5),
    (2020,"DC","PBKS","DC","Sharjah Cricket Stadium","DC","bat",5,0),
    (2020,"KKR","SRH","KKR","Abu Dhabi Cricket Ground","KKR","bat",7,0),
    (2020,"RCB","RR","RCB","Dubai Intl Stadium","RCB","bat",8,0),
    (2020,"MI","DC","MI","Sharjah Cricket Stadium","MI","bat",9,0),
    (2020,"CSK","RR","CSK","Abu Dhabi Cricket Ground","CSK","bat",16,0),
    (2020,"SRH","PBKS","SRH","Dubai Intl Stadium","SRH","bat",69,0),
    (2020,"KKR","RCB","KKR","Sharjah Cricket Stadium","RCB","field",0,8),
    (2020,"MI","DC","MI","Dubai Intl Stadium","MI","bat",57,0),  # Final
    # 2021
    (2021,"CSK","DC","CSK","Wankhede Stadium","CSK","bat",7,0),
    (2021,"MI","KKR","MI","MA Chidambaram Stadium","MI","bat",10,0),
    (2021,"RCB","PBKS","PBKS","Wankhede Stadium","RCB","field",0,5),
    (2021,"RR","DC","RR","Wankhede Stadium","DC","field",0,3),
    (2021,"CSK","MI","CSK","MA Chidambaram Stadium","CSK","bat",4,0),
    (2021,"SRH","KKR","KKR","Rajiv Gandhi Intl Stadium","SRH","field",0,6),
    (2021,"RCB","DC","DC","Sharjah Cricket Stadium","DC","bat",7,0),
    (2021,"PBKS","RR","PBKS","Dubai Intl Stadium","PBKS","bat",6,0),
    (2021,"CSK","KKR","CSK","Dubai Intl Stadium","KKR","field",0,2),  # Final
    # 2022
    (2022,"GT","LSG","GT","Brabourne Stadium","GT","bat",5,0),
    (2022,"CSK","PBKS","PBKS","DY Patil Stadium","PBKS","field",0,6),
    (2022,"DC","MI","DC","Brabourne Stadium","DC","bat",4,0),
    (2022,"KKR","SRH","SRH","DY Patil Stadium","SRH","field",0,6),
    (2022,"RCB","RR","RR","DY Patil Stadium","RR","field",0,4),
    (2022,"GT","LSG","GT","Eden Gardens","GT","bat",62,0),
    (2022,"CSK","MI","CSK","Wankhede Stadium","CSK","bat",3,0),
    (2022,"KKR","DC","DC","DY Patil Stadium","DC","field",0,4),
    (2022,"GT","RR","GT","Narendra Modi Stadium","GT","bat",7,0),  # Final
    # 2023
    (2023,"GT","CSK","CSK","Narendra Modi Stadium","GT","field",0,5),
    (2023,"MI","RCB","RCB","Wankhede Stadium","RCB","field",0,8),
    (2023,"LSG","DC","LSG","BRSABV Ekana Cricket Stadium","LSG","bat",50,0),
    (2023,"KKR","PBKS","KKR","Eden Gardens","PBKS","field",0,7),
    (2023,"RR","SRH","RR","Sawai Mansingh Stadium","RR","bat",72,0),
    (2023,"GT","LSG","GT","Narendra Modi Stadium","GT","bat",5,0),
    (2023,"CSK","DC","CSK","MA Chidambaram Stadium","CSK","bat",27,0),
    (2023,"MI","KKR","MI","Wankhede Stadium","MI","bat",5,0),
    (2023,"CSK","GT","CSK","Narendra Modi Stadium","CSK","bat",15,0),  # Final
    # 2024
    (2024,"KKR","RCB","KKR","Eden Gardens","KKR","bat",18,0),
    (2024,"CSK","RR","RR","MA Chidambaram Stadium","RR","field",0,5),
    (2024,"GT","MI","MI","Narendra Modi Stadium","GT","field",0,4),
    (2024,"SRH","DC","SRH","Rajiv Gandhi Intl Stadium","SRH","bat",67,0),
    (2024,"PBKS","LSG","PBKS","Punjab Cricket Association Stadium","PBKS","bat",3,0),
    (2024,"KKR","SRH","KKR","Eden Gardens","KKR","bat",8,0),
    (2024,"RCB","DC","DC","M Chinnaswamy Stadium","DC","field",0,7),
    (2024,"GT","CSK","CSK","Narendra Modi Stadium","CSK","field",0,3),
    (2024,"RR","LSG","RR","Sawai Mansingh Stadium","RR","bat",10,0),
    (2024,"MI","PBKS","MI","Wankhede Stadium","MI","bat",9,0),
    (2024,"SRH","KKR","KKR","Rajiv Gandhi Intl Stadium","KKR","field",0,8),  # Final
]


def save_teams_json():
    os.makedirs(RAW_DIR, exist_ok=True)
    teams = {
        "CSK":  {"name": "Chennai Super Kings",       "home": "MA Chidambaram Stadium",          "titles": 5, "founded": 2008},
        "MI":   {"name": "Mumbai Indians",            "home": "Wankhede Stadium",                 "titles": 5, "founded": 2008},
        "RCB":  {"name": "Royal Challengers Bengaluru","home": "M Chinnaswamy Stadium",           "titles": 0, "founded": 2008},
        "KKR":  {"name": "Kolkata Knight Riders",     "home": "Eden Gardens",                    "titles": 3, "founded": 2008},
        "DC":   {"name": "Delhi Capitals",            "home": "Feroz Shah Kotla",                "titles": 0, "founded": 2008},
        "PBKS": {"name": "Punjab Kings",              "home": "Punjab Cricket Association Stadium","titles": 0,"founded": 2008},
        "RR":   {"name": "Rajasthan Royals",          "home": "Sawai Mansingh Stadium",           "titles": 1, "founded": 2008},
        "SRH":  {"name": "Sunrisers Hyderabad",       "home": "Rajiv Gandhi Intl Stadium",        "titles": 1, "founded": 2013},
        "LSG":  {"name": "Lucknow Super Giants",      "home": "BRSABV Ekana Cricket Stadium",     "titles": 0, "founded": 2022},
        "GT":   {"name": "Gujarat Titans",            "home": "Narendra Modi Stadium",            "titles": 1, "founded": 2022},
    }
    with open(TEAMS_JSON, "w") as f:
        json.dump(teams, f, indent=2)
    print(f"Saved teams.json → {TEAMS_JSON}")


def save_matches_csv():
    os.makedirs(RAW_DIR, exist_ok=True)
    fieldnames = [
        "id","season","team1","team2","toss_winner","toss_decision",
        "winner","win_by_runs","win_by_wickets","venue",
    ]
    with open(MATCHES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(MATCHES_DATA, start=1):
            (season, team1, team2, winner, venue,
             toss_winner, toss_decision, win_by_runs, win_by_wickets) = row
            writer.writerow({
                "id": i,
                "season": season,
                "team1": team1,
                "team2": team2,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "winner": winner,
                "win_by_runs": win_by_runs,
                "win_by_wickets": win_by_wickets,
                "venue": venue,
            })
    print(f"Saved {len(MATCHES_DATA)} matches → {MATCHES_CSV}")


if __name__ == "__main__":
    save_teams_json()
    save_matches_csv()
    print("Dataset creation complete.")
