"""Project-wide constants and filesystem paths.

Pure stdlib (no third-party imports) so it can be imported anywhere, including by
the reference-data layer, without installing the ML/runtime dependencies.
"""
from pathlib import Path

# --- Filesystem layout ---
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../Brawlstars-Draft-Tool
BACKEND_DIR = REPO_ROOT / "backend"
DATA_DIR = REPO_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# --- Brawl Stars ranked ---
# Ranked rotates its mode set each season. This reflects the modes currently observed
# in live ranked (soloRanked/teamRanked) battle data; update when the rotation changes.
RANKED_MODES = ("Gem Grab", "Brawl Ball", "Knockout", "Hot Zone", "Heist", "Bounty")

# Battle logs use camelCase mode names; reference data uses display names.
MODE_CAMEL_TO_DISPLAY = {
    "gemGrab": "Gem Grab",
    "brawlBall": "Brawl Ball",
    "knockout": "Knockout",
    "hotZone": "Hot Zone",
    "heist": "Heist",
    "bounty": "Bounty",
}

# Official 7-class taxonomy (Supercell)
BRAWLER_CLASSES = (
    "Tank",
    "Assassin",
    "Controller",
    "Marksman",
    "Support",
    "Damage Dealer",
    "Artillery",
)
UNCLASSIFIED = "Unclassified"

# --- Draft format ---
TEAM_SIZE = 3
NUM_BANS_PER_TEAM = 3
# 1-2-2-1 snake: which team (0 = first-pick team, 1 = second) picks at each of the 6 steps
PICK_ORDER = (0, 1, 1, 0, 0, 1)
PICK_SECONDS = 20
