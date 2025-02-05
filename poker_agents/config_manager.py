import json
import os
from pathlib import Path
from typing import Dict, Optional

class ConfigManager:
    def __init__(self, config_file: Path):
        self.config_file = config_file

    def load_config(self) -> dict:
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return self._get_env_config()

    def _get_env_config(self) -> dict:
        return {
            'contracts': {
                'router': os.getenv('POKER_ROUTER_ADDRESS'),
                'state_storage': os.getenv('POKER_STATE_STORAGE_ADDRESS'),
                'game_logic': os.getenv('POKER_GAME_LOGIC_ADDRESS')
            },
            'timer_agent': {
                'rpc_url': os.getenv('POKER_TIMER_RPC'),
                'private_key': os.getenv('POKER_TIMER_KEY')
            },
            'agents': {}
        }

    def save_agent_config(self, name: str, rpc_env: str, key_env: str, model_env: str = None):
        config = self.load_config()
        config['agents'][name] = {
            'rpc_url': rpc_env,
            'private_key': key_env,
            'model': model_env
        }
        self.save_config(config)

    def save_config(self, config: dict):
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)