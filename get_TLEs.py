from operator import index
import os
import json
import requests
import argparse
import pandas as pd


GROUPS = ["stations","weather","resource","tdrss","argos","geo","satnogs","gnss","sbass","nnss","musson","science","education","cubesat","military","gps-ops","amateur"]

def get_TLEs_from_API(groups = GROUPS):
    TLEs = []
    for group in groups:
        query = {'GROUP':group,'FORMAT':'JSON'}
        response = requests.get('https://www.celestrak.com/NORAD/elements/gp.php', params=query)
        groupTLEs = pd.DataFrame.from_dict(response.json())
        groupTLEs["GROUP"] = group
        groupTLEs["downloaded"] = pd.Timestamp.now()
        TLEs.extend([groupTLEs])

    TLEs = pd.concat(TLEs)

    return TLEs.drop_duplicates(subset=TLEs.columns.difference(['GROUP',"Dowenloaded"]))


def main():

    parser = argparse.ArgumentParser(
        description="Download TLE data from www.celestrak.com API."
    )
    parser.add_argument("--output-file", default="TLEs.csv",  help = "output JSON file where TLEs will be saved")
    args = parser.parse_args()

    TLE_output_file = args.output_file
    TLEs = get_TLEs_from_API(groups = GROUPS)
    TLEs.reset_index().to_csv(TLE_output_file,index = False)

if __name__ == "__main__":
    main()