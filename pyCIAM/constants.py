# Retreat cases
RLIST = ["retreat1", "retreat10", "retreat100", "retreat1000", "retreat10000"]

# Protect cases
PLIST = ["protect10", "protect100", "protect1000", "protect10000"]

# All adaptation cases to calculate
SOLVCASES = ["noAdaptation"] + PLIST + RLIST  # cases to calc
CASES = SOLVCASES + ["optimalfixed"]  # all cases + slot for optimal (least cost) case

# data dictionary for different adaptation cases
CASE_DICT = {k: kx for kx, k in enumerate(SOLVCASES)}

# Cost types for which damages are calculated in pyCIAM()
COSTTYPES = [
    "wetland",
    "inundation",
    "relocation",
    "protection",
    "stormCapital",
    "stormPopulation",
]
