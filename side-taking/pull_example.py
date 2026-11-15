"""Smallest runnable example: how far does the rating follow the user in one dispute?
Run from this folder:  python pull_example.py"""
import pandas as pd

user = pd.read_csv("data/user_no_thinking_first.csv", dtype={"scenario_id": str})   # user's view stated first
s = "028"                                   # the bakery dispute; position A is the positive end
a = user[(user.scenario_id == s) & (user.narrator_side == "A")].rating.mean()
b = user[(user.scenario_id == s) & (user.narrator_side == "B")].rating.mean()
print("mean rating when the user holds A:", round(a, 2))
print("mean rating when the user holds B:", round(b, 2))
print("pull toward the user:", round((a - b) / 2, 2))