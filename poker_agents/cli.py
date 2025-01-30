import questionary
import asyncio
import json
from pathlib import Path
import logging
from getpass import getpass
import os
from typing import Dict
from tabulate import tabulate
from datetime import datetime

from .agent import PokerAgent
from .constants import CONFIG_DIR, PROFILES_FILE, CURRENT_SESSION, SessionConfig
from .openrouter_client import OpenRouterClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfileManager:
    def __init__(self):
        self.ensure_config_dir()

    def ensure_config_dir(self):
        """Ensure configuration directory exists"""
        CONFIG_DIR.mkdir(exist_ok=True)

    def load_profiles(self) -> dict:
        """Load existing agent profiles"""
        if PROFILES_FILE.exists():
            with open(PROFILES_FILE, 'r') as f:
                return json.load(f)
        return {'profiles': {}}

    def save_profiles(self, profiles: dict):
        """Save agent profiles"""
        with open(PROFILES_FILE, 'w') as f:
            json.dump(profiles, f, indent=2)

    def create_profile(self, name: str, rpc_url: str, private_key: str, 
                      model_name: str) -> bool:
        """Create a new agent profile"""
        profiles = self.load_profiles()
        
        if name in profiles['profiles']:
            return False

        profiles['profiles'][name] = {
            'id': str(datetime.now().timestamp()),
            'rpc_url': rpc_url,
            'private_key': private_key,  # Note: In production, encrypt this
            'model_name': model_name,
            'created_at': datetime.now().isoformat(),
            'last_used': None
        }
        
        self.save_profiles(profiles)
        return True

    def get_profile(self, name: str) -> dict:
        """Get a specific profile"""
        profiles = self.load_profiles()
        return profiles['profiles'].get(name)

    def update_last_used(self, name: str):
        """Update last used timestamp"""
        profiles = self.load_profiles()
        if name in profiles['profiles']:
            profiles['profiles'][name]['last_used'] = datetime.now().isoformat()
            self.save_profiles(profiles)

class PokerAgentCLI:
    def __init__(self):
        self.profile_manager = ProfileManager()
        self.active_agents: Dict[str, PokerAgent] = {}
        self.running = True

    async def cleanup(self):
        """Clean up all running agents"""
        if self.active_agents:
            print("\nStopping all active agents...")
            await self.stop_all_agents()
        self.running = False

    async def main_menu(self):
        """Main menu loop with error handling"""
        try:
            # First, set up contract addresses for the session
            await self.setup_contract_addresses()

            while self.running:
                try:
                    choice = await questionary.select(
                        "Poker Agent Manager - Main Menu",
                        choices=[
                            "1. Manage Profiles",
                            "2. Start Agents",
                            "3. Stop Agents",
                            "4. View Status",
                            "5. Update Contract Addresses",
                            "6. Exit"
                        ]
                    ).ask_async()

                    if "1." in choice:
                        await self.profile_management_menu()
                    elif "2." in choice:
                        await self.start_agents_menu()
                    elif "3." in choice:
                        await self.stop_agents_menu()
                    elif "4." in choice:
                        await self.view_status()
                    elif "5." in choice:
                        await self.setup_contract_addresses()
                    else:
                        if await self.confirm_exit():
                            break

                except Exception as e:
                    logger.error(f"Error in menu operation: {e}")
                    print(f"\nError occurred: {e}")
                    if await questionary.confirm("Continue to main menu?").ask_async():
                        continue
                    else:
                        break

        except KeyboardInterrupt:
            print("\nReceived exit signal...")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            print(f"\nFatal error occurred: {e}")
        finally:
            await self.cleanup()

    async def setup_contract_addresses(self):
        """Set up contract addresses for the session"""
        print("\nSet up contract addresses for this session")
        print("(These will be used for all agents until you exit)")
        
        CURRENT_SESSION.router_address = await questionary.text(
            "Enter Router contract address:",
            default=CURRENT_SESSION.router_address or ""
        ).ask_async()
        
        CURRENT_SESSION.state_storage_address = await questionary.text(
            "Enter StateStorage contract address:",
            default=CURRENT_SESSION.state_storage_address or ""
        ).ask_async()

        CURRENT_SESSION.game_logic_address = await questionary.text(
            "Enter GameLogic contract address:",
            default=CURRENT_SESSION.game_logic_address or ""
        ).ask_async()

    async def profile_management_menu(self):
        """Profile management menu"""
        while True:
            choice = await questionary.select(
                "Profile Management",
                choices=[
                    "1. Create New Profile",
                    "2. View Profiles",
                    "3. Delete Profile",
                    "4. Back to Main Menu"
                ]
            ).ask_async()

            if "1." in choice:
                await self.create_profile()
            elif "2." in choice:
                await self.view_profiles()
            elif "3." in choice:
                await self.delete_profile()
            else:
                break

    async def create_profile(self):
        """Create new agent profile"""
        try:
            if not os.getenv("OPENROUTER_API_KEY"):
                print("OpenRouter API key not found! Please set OPENROUTER_API_KEY environment variable.")
                return

            # Get available models
            print("Fetching available models from OpenRouter...")
            client = OpenRouterClient()
            models = client.get_available_models()
            
            if not models:
                print("Failed to fetch models. Using default (claude-2)")
                model_name = "anthropic/claude-2"
            else:
                # Format choices with model name and context length
                model_choices = [
                    f"{model['id']} ({model['context_length']} tokens)"
                    for model in models
                ]
                choice = await questionary.select(
                    "Select model to use:",
                    choices=model_choices
                ).ask_async()
                # Extract model ID from choice
                model_name = choice.split(" (")[0]

            # Get profile info
            name = await questionary.text("Enter profile name:").ask_async()
            
            if self.profile_manager.get_profile(name):
                print(f"Profile '{name}' already exists!")
                return

            rpc_url = await questionary.text("Enter RPC URL:").ask_async()
            private_key = await questionary.password(
                "Enter private key (will be encrypted):"
            ).ask_async()

            success = self.profile_manager.create_profile(
                name=name,
                rpc_url=rpc_url,
                private_key=private_key,
                model_name=model_name
            )

            if success:
                print(f"Profile '{name}' created successfully!")
                print(f"Model: {model_name}")
                print(f"RPC URL: {rpc_url}")
            else:
                print(f"Failed to create profile '{name}'")

        except Exception as e:
            print(f"Error creating profile: {e}")
            logger.error(f"Profile creation error: {e}")

    async def view_profiles(self):
        """Display all profiles"""
        profiles = self.profile_manager.load_profiles()['profiles']
        
        if not profiles:
            print("No profiles found.")
            return

        headers = ['Name', 'Model', 'RPC URL', 'Created', 'Last Used', 'Status']
        rows = []

        for name, profile in profiles.items():
            status = 'Running' if profile['id'] in self.active_agents else 'Stopped'
            rows.append([
                name,
                profile['model_name'],
                profile['rpc_url'],
                profile['created_at'][:10],
                profile['last_used'][:10] if profile['last_used'] else 'Never',
                status
            ])

        print("\nAgent Profiles:")
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        input("\nPress Enter to continue...")

    async def delete_profile(self):
        """Delete a profile"""
        profiles = self.profile_manager.load_profiles()['profiles']
        
        if not profiles:
            print("No profiles to delete!")
            return

        name = await questionary.select(
            "Select profile to delete:",
            choices=list(profiles.keys()) + ["Cancel"]
        ).ask_async()

        if name == "Cancel":
            return

        if profiles[name]['id'] in self.active_agents:
            print(f"Cannot delete profile '{name}' while agent is running!")
            return

        # Delete profile
        profiles.pop(name)
        self.profile_manager.save_profiles({'profiles': profiles})
        print(f"Profile '{name}' deleted successfully!")

    async def start_agents_menu(self):
        """Menu for starting agents"""
        profiles = self.profile_manager.load_profiles()['profiles']
        if not profiles:
            print("No profiles found. Please create a profile first.")
            return

        choices = [f"Start {name}" for name in profiles.keys()]
        choices.append("Start Multiple Agents")
        choices.append("Back")

        choice = await questionary.select(
            "Select agent to start:",
            choices=choices
        ).ask_async()

        if choice == "Back":
            return
        elif choice == "Start Multiple Agents":
            await self.start_multiple_agents()
        else:
            name = choice.replace("Start ", "")
            await self.start_single_agent(name)

    async def start_single_agent(self, name: str):
        """Start a single agent"""
        try:
            # Check contract addresses
            if not all([
                CURRENT_SESSION.router_address,
                CURRENT_SESSION.state_storage_address,
                CURRENT_SESSION.game_logic_address
            ]):
                print("Contract addresses not set! Please update them first.")
                return

            # Get and validate profile
            profile = self.profile_manager.get_profile(name)
            if not profile:
                print(f"Profile '{name}' not found!")
                return

            if profile['id'] in self.active_agents:
                print(f"Agent '{name}' is already running!")
                return

            print(f"Starting agent '{name}'...")
            
            agent = PokerAgent(model_name=profile['model_name'])
            try:
                success = await agent.initialize(
                    rpc_url=profile['rpc_url'],
                    private_key=profile['private_key'],
                    router_address=CURRENT_SESSION.router_address,
                    state_storage_address=CURRENT_SESSION.state_storage_address,
                    game_logic_address=CURRENT_SESSION.game_logic_address
                )

                if success:
                    self.active_agents[profile['id']] = agent
                    self.profile_manager.update_last_used(name)
                    task = asyncio.create_task(self._run_agent(name, profile['id'], agent))
                    # Add error callback
                    task.add_done_callback(lambda t: self._handle_agent_completion(t, name, profile['id']))
                    print(f"Agent '{name}' started successfully!")
                else:
                    print(f"Failed to initialize agent '{name}'")

            except Exception as e:
                print(f"Error starting agent '{name}': {e}")
                logger.error(f"Agent start error: {e}", exc_info=True)
                if agent in self.active_agents.values():
                    await agent.stop()
                    if profile['id'] in self.active_agents:
                        del self.active_agents[profile['id']]

        except Exception as e:
            print(f"Unexpected error starting agent '{name}': {e}")
            logger.error(f"Unexpected agent start error: {e}", exc_info=True)
            return False

    def _handle_agent_completion(self, task, name: str, profile_id: str):
        """Handle agent task completion"""
        try:
            task.result()  # Will raise exception if task failed
        except asyncio.CancelledError:
            logger.info(f"Agent '{name}' was cancelled")
        except Exception as e:
            logger.error(f"Agent '{name}' failed with error: {e}")
        finally:
            if profile_id in self.active_agents:
                del self.active_agents[profile_id]
            print(f"Agent '{name}' stopped")

    async def _run_agent(self, name: str, profile_id: str, agent: PokerAgent):
        """Run agent and handle cleanup on completion"""
        try:
            await agent.start()
        except Exception as e:
            logger.error(f"Agent '{name}' error: {e}", exc_info=True)
            print(f"Agent '{name}' encountered an error: {e}")
            raise  # Re-raise to trigger error handling
        finally:
            if profile_id in self.active_agents:
                del self.active_agents[profile_id]
            print(f"Agent '{name}' stopped")

    async def start_multiple_agents(self):
        """Start multiple selected agents"""
        profiles = self.profile_manager.load_profiles()['profiles']
        available_agents = [
            name for name, profile in profiles.items()
            if profile['id'] not in self.active_agents
        ]

        if not available_agents:
            print("No available agents to start!")
            return

        selected = await questionary.checkbox(
            "Select agents to start:",
            choices=available_agents
        ).ask_async()

        if not selected:
            return

        for name in selected:
            await self.start_single_agent(name)

    async def stop_agents_menu(self):
        """Menu for stopping agents"""
        if not self.active_agents:
            print("No agents are currently running.")
            return

        profiles = self.profile_manager.load_profiles()['profiles']
        choices = [
            f"Stop {name}" for name, profile in profiles.items()
            if profile['id'] in self.active_agents
        ]
        choices.extend(["Stop All Agents", "Back"])

        choice = await questionary.select(
            "Select agent to stop:",
            choices=choices
        ).ask_async()

        if choice == "Back":
            return
        elif choice == "Stop All Agents":
            await self.stop_all_agents()
        else:
            name = choice.replace("Stop ", "")
            await self.stop_single_agent(name)

    async def stop_single_agent(self, name: str):
        """Stop a single agent"""
        profile = self.profile_manager.get_profile(name)
        if not profile or profile['id'] not in self.active_agents:
            print(f"Agent '{name}' is not running!")
            return

        await self.active_agents[profile['id']].stop()
        del self.active_agents[profile['id']]
        print(f"Agent '{name}' stopped successfully!")

    async def stop_all_agents(self):
        """Stop all running agents"""
        if not self.active_agents:
            print("No agents are currently running!")
            return

        for agent in self.active_agents.values():
            await agent.stop()
        
        self.active_agents.clear()
        print("All agents stopped successfully!")

    async def view_status(self):
        """View status of all agents"""
        profiles = self.profile_manager.load_profiles()['profiles']
        
        headers = ['Name', 'Status', 'Model', 'Last Action']
        rows = []

        for name, profile in profiles.items():
            status = 'Running' if profile['id'] in self.active_agents else 'Stopped'
            last_action = 'Never'
            if status == 'Running':
                agent = self.active_agents[profile['id']]
                if agent.last_action_time:
                    last_action = agent.last_action_time.strftime('%H:%M:%S')

            rows.append([
                name,
                status,
                profile['model_name'],
                last_action
            ])

        print("\nAgent Status:")
        print(tabulate(rows, headers=headers, tablefmt='grid'))
        input("\nPress Enter to continue...")

    async def confirm_exit(self):
        """Confirm exit if agents are running"""
        if self.active_agents:
            return await questionary.confirm(
                "There are still agents running. Are you sure you want to exit?"
            ).ask_async()
        return True

async def main():
    cli = PokerAgentCLI()
    try:
        await cli.main_menu()
    except KeyboardInterrupt:
        if cli.active_agents:
            print("\nStopping all agents...")
            await cli.stop_all_agents()
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if cli.active_agents:
            await cli.stop_all_agents()

if __name__ == "__main__":
    asyncio.run(main())