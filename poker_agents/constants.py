from enum import IntEnum
from pathlib import Path
from dataclasses import dataclass

class BettingRound(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

@dataclass
class SessionConfig:
    router_address: str = None
    state_storage_address: str = None
    game_logic_address: str = None

# Global session config that will be used by all agents
CURRENT_SESSION = SessionConfig()
CONFIG_FILE = CONFIG_DIR / 'config.json'
CONFIG_DIR = Path.home() / '.poker-agent'
PROFILES_FILE = CONFIG_DIR / 'profiles.json'
MIN_BALANCE_THRESHOLD = 0.1