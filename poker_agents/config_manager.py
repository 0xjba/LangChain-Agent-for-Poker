import json
import os
from pathlib import Path
from typing import Dict, Optional

class ConfigManager:
    def __init__(self, config_file: Path):
        self.config_file = config_file

    def load_config(self) -> dict:
        config = None
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                try:
                    # First load raw config to see what's there
                    raw_config = json.load(f)
                    print(f"Raw config agents: {raw_config.get('agents', {})}")
                    
                    config = raw_config
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    return self._get_env_config()
        else:
            config = self._get_env_config()
        
        # Resolve environment variables in the contracts section
        if 'contracts' in config:
            for key, value in config['contracts'].items():
                if isinstance(value, str):
                    env_value = os.getenv(value)
                    if env_value:
                        config['contracts'][key] = env_value
        
        # Resolve environment variables in the timer_agent section
        if 'timer_agent' in config:
            for key, value in config['timer_agent'].items():
                if isinstance(value, str):
                    env_value = os.getenv(value)
                    if env_value:
                        config['timer_agent'][key] = env_value
        
        # Preserve agent entries even if environment variables can't be resolved
        if 'agents' in config:
            print(f"Processing {len(config['agents'])} agents:")
            for agent_name, agent_config in list(config['agents'].items()):  # Use list() to avoid modification during iteration
                print(f"  Agent: {agent_name}")
                has_required_fields = True
                
                # Resolve environment variables for this agent
                for key, value in agent_config.items():
                    if isinstance(value, str):
                        env_value = os.getenv(value)
                        if env_value:
                            print(f"    Resolved {key}: {value} -> {env_value}")
                            agent_config[key] = env_value
                        else:
                            print(f"    Could not resolve {key}: {value} (keeping original)")
                            # Keep the original value
        
        print(f"Final config agents: {config.get('agents', {})}")
        return config

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