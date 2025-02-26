from enum import IntEnum
from pathlib import Path
from dataclasses import dataclass
import os

class BettingRound(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

# Find the project directory (where the script is running from)
PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Global session config that will be used by all agents
CONFIG_DIR = Path.home() / '.poker-agent'  # Keep this for backward compatibility
PROFILES_FILE = CONFIG_DIR / 'profiles.json'  # Keep profiles in the home directory
CONFIG_FILE = PROJECT_DIR / 'config.json'  # Look for config.json in the project directory
MIN_BALANCE_THRESHOLD = 0.1

@dataclass
class SessionConfig:
    router_address: str = None
    state_storage_address: str = None
    game_logic_address: str = None

CURRENT_SESSION = SessionConfig()