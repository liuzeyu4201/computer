# research itunes library
import json
import requests
import sys

def get_songs():
    try:
        limit=int(input("limit search number:"))
    except:
        print(" not a number")
    term=input("input term name:")
    return requests.get("https://itunes.apple.com/search?entity=song&limit={limit}&term={term}").json()

