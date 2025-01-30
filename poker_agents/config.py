import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

class ProfileManager:
    def __init__(self, config_dir: Path, profiles_file: Path):
        self.config_dir = config_dir
        self.profiles_file = profiles_file
        self.ensure_config_dir()

    def ensure_config_dir(self):
        """Ensure configuration directory exists"""
        self.config_dir.mkdir(exist_ok=True)

    def load_profiles(self) -> dict:
        """Load existing agent profiles"""
        if self.profiles_file.exists():
            with open(self.profiles_file, 'r') as f:
                return json.load(f)
        return {'profiles': {}}

    def save_profiles(self, profiles: dict):
        """Save agent profiles"""
        with open(self.profiles_file, 'w') as f:
            json.dump(profiles, f, indent=2)

    def create_profile(self, name: str, rpc_url: str, private_key: str) -> bool:
        """Create a new agent profile with private key"""
        profiles = self.load_profiles()
        
        if name in profiles['profiles']:
            return False
        
        # Encrypt private key before storing (using simple encryption for example)
        encrypted_key = self._encrypt_private_key(private_key)
        
        profiles['profiles'][name] = {
            'id': str(datetime.now().timestamp()),
            'rpc_url': rpc_url,
            'private_key': encrypted_key,  # Store encrypted private key
            'created_at': datetime.now().isoformat(),
            'last_used': None
        }
        
        self.save_profiles(profiles)
        return True

    def _encrypt_private_key(self, private_key: str) -> str:
        """Simple encryption for private key - replace with proper encryption"""
        # This is a placeholder - implement proper encryption
        return private_key  # In real implementation, encrypt this!

    def _decrypt_private_key(self, encrypted_key: str) -> str:
        """Simple decryption for private key - replace with proper decryption"""
        # This is a placeholder - implement proper decryption
        return encrypted_key  # In real implementation, decrypt this!

    def get_private_key(self, profile_name: str) -> Optional[str]:
        """Get decrypted private key for a profile"""
        profile = self.get_profile(profile_name)
        if profile and 'private_key' in profile:
            return self._decrypt_private_key(profile['private_key'])
        return None

    def update_last_used(self, name: str):
        """Update last used timestamp for a profile"""
        profiles = self.load_profiles()
        if name in profiles['profiles']:
            profiles['profiles'][name]['last_used'] = datetime.now().isoformat()
            self.save_profiles(profiles)

    def delete_profile(self, name: str) -> bool:
        """Delete an agent profile"""
        profiles = self.load_profiles()
        if name not in profiles['profiles']:
            return False
        
        del profiles['profiles'][name]
        self.save_profiles(profiles)
        return True

    def get_profile(self, name: str) -> Optional[dict]:
        """Get a specific profile"""
        profiles = self.load_profiles()
        return profiles['profiles'].get(name)