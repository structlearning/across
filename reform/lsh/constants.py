LAMBDA_FENCE_SIT = 0.1
# LAMBDA_COLL_MIN_VALUES = [
#     0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
#     0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
#     0.75, 0.80, 0.85,
# ]
LAMBDA_COLL_MIN_VALUES = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
]
LINEAR_EVERY_DIM_VALUES = [False]
SAME_IDXS_ACROSS_DIMS_VALUES = [False, True]
SUBSET_SIZES = list(range(21))  # 0 to 20
TOP_KS = [5, 10, 20, 50, 100, 500, 1000, 5000, 10000]

LSH_SEED = 42
MIN_FREE_MB = 2000

# Seconds between polling cycles.
POLL_INTERVAL = 60

# How many times to retry a failed run before giving up.
MAX_RETRIES = 20