from web3 import Web3
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class TimerAgent:
    def __init__(self):
        self.is_running = False
        self.web3 = None
        self.account = None
        self.router = None
        self.game_logic = None
        self.active_timers: Dict[str, datetime] = {}

    async def initialize(self, rpc_url: str, private_key: str, 
                        router_address: str, game_logic_address: str) -> bool:
        """Initialize timer agent with contracts"""
        try:
            # Initialize Web3 and account
            self.web3 = Web3(Web3.HTTPProvider(rpc_url))
            self.account = self.web3.eth.account.from_key(private_key)
            
            # Load contract ABIs
            with open('abis/Router.json', 'r') as f:
                router_abi = json.load(f)
            with open('abis/GameLogic.json', 'r') as f:
                game_logic_abi = json.load(f)

            # Initialize contracts
            self.router = self.web3.eth.contract(
                address=router_address,
                abi=router_abi
            )
            
            self.game_logic = self.web3.eth.contract(
                address=game_logic_address,
                abi=game_logic_abi
            )

            # Check if this account is authorized - don't use await here
            is_authorized = self.router.functions.isAuthorizedTimer(
                self.account.address
            ).call()
            
            if not is_authorized:
                logger.error(f"Timer agent {self.account.address} is not authorized!")
                return False

            logger.info(f"Timer agent initialized with address: {self.account.address}")
            return True

        except Exception as e:
            logger.error(f"Timer initialization failed: {e}")
            return False

    async def monitor_timers(self):
        """Monitor active timers and handle timeouts"""
        while self.is_running:
            try:
                current_time = datetime.now()
                expired_players = [
                    player for player, expiry in self.active_timers.items()
                    if current_time >= expiry
                ]

                for player in expired_players:
                    await self.process_timeout(player)
                    del self.active_timers[player]

                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"Error monitoring timers: {e}")
                await asyncio.sleep(1)

    async def process_timeout(self, player_address: str):
        """Process a player timeout"""
        try:
            # Build transaction
            tx = self.router.functions.routeTimeoutAction(
                player_address
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.web3.eth.get_transaction_count(self.account.address),
                'gas': 200000,
                'gasPrice': self.web3.eth.gas_price
            })

            # Sign and send transaction
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction receipt
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt['status'] == 1:
                logger.info(f"Timeout processed for player: {player_address}")
            else:
                logger.error(f"Timeout transaction failed for player: {player_address}")

        except Exception as e:
            logger.error(f"Error processing timeout for {player_address}: {e}")

    async def monitor_events(self):
        """Monitor contract events for turns using getLogs"""
        while self.is_running:
            try:
                # Get latest block number
                latest_block = self.web3.eth.block_number
                from_block = max(0, latest_block - 10)  # Look back 10 blocks

                # Event signature for ActionTimerStarted
                event_signature = '0x' + self.web3.keccak(
                    text="ActionTimerStarted(address,uint256,uint256)"
                ).hex()
                event_signature = self.web3.to_hex(hexstr=event_signature)
                
                # Get logs directly
                logs = self.web3.eth.get_logs({
                    'address': self.game_logic.address,
                    'fromBlock': from_block,
                    'toBlock': 'latest',
                    'topics': [
                        event_signature,
                        '0x' + self.account.address[2:].zfill(64)  # Ensure 0x prefix
                    ]
                })

                # Process any logs found
                for log in logs:
                    # Process the log data
                    topics = log['topics']
                    player_address = self.web3.to_checksum_address(topics[1][-40:])  # last 20 bytes
                    if player_address == self.account.address:
                        logger.info(f"Turn event detected: Block {log['blockNumber']}")
                        await self.process_turn()

            except Exception as e:
                logger.error(f"Error monitoring events: {e}")
            
            await asyncio.sleep(1)

    async def start(self):
        """Start the timer agent"""
        self.is_running = True
        logger.info("Starting timer agent...")
        
        try:
            # Run both monitoring methods concurrently
            await asyncio.gather(
                self.monitor_events(),
                self.monitor_timers()
            )
        except Exception as e:
            logger.error(f"Error in timer agent main loop: {e}")
            self.is_running = False

    async def stop(self):
        """Stop the timer agent"""
        self.is_running = False
        logger.info("Stopping timer agent...")
        self.active_timers.clear()