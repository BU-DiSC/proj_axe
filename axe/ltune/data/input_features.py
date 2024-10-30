kSYSTEM_HEADER = [
    "entry_p_page",
    "selec",
    "entry_size",
    "max_h",
    "num_elem",
    "rho",
]

kHEADER_ROBUST = [
    "entry_p_page",
    "selec",
    "entry_size",
    "max_h",
    "num_elem",
    "rho",
]

kWORKLOAD_HEADER = [
    "z0",
    "z1",
    "q",
    "w",
]

kINPUT_FEATS = kWORKLOAD_HEADER + kSYSTEM_HEADER
kINPUT_FEATS_ROBUST = kWORKLOAD_HEADER + kHEADER_ROBUST
