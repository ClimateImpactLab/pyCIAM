"""This module contains various constants which reference lists of string values
defining cost types and potential adaptation options.

Constants
    RLIST : list of str
        List of the discrete retreat options, corresponding to different return heights
        of extreme sea level. Other than ``1``, the suffixes in this list must be included
        in the `surge_heights` coordinates of SLIIDERS (or a similarly formatted
        dataset).
    PLIST : list of str
        Same as RLIST except for protection. pyCIAM does not allow for protection to the
        maximum projected local MSL height during an adaptation period, as seen with
        `retreat1` in `RLIST`.
    SOLVCASES : list of str
        A list of all adaptation options available to decision-making coastal segment
        agents, including the "reactive retreat" option (`noAdaptation`).
    CASES : list of str
        Same as SOLVCASES, but including the `optimalfixed` case, which is equivalent
        to one of the other `CASES`. It represents the optimal adaptation choice,
        which will be segment-specific.
    CASE_DICT : dict
        Keys are values of `SOLVCASES`. Values are integer index values.
    COSTTYPES : list of str
        A list of all cost types calculated in pyCIAM.
"""

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
