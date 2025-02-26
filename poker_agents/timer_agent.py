from web3 import Web3
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class TimerAgent:
    def __init__(self):
        # Existing initialization
        self.is_running = False
        self.web3 = None
        self.account = None
        self.router = None
        self.game_logic = None
        self.active_timers = {}
        self.betting_round_monitor = False
        self.current_round = None
        self.last_round_check = datetime.now()
        self.ROUND_CHECK_INTERVAL = 2
        
        # Add persistsent tracking of processed blocks
        self.last_processed_block = 0
        self.processed_events = set()  # Set of event IDs we've already processed

    async def initialize(self, rpc_url: str, private_key: str, 
                        router_address: str, game_logic_address: str, 
                        state_storage_address: str = None) -> bool:
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
            
            # Initialize StateStorage contract - CRITICAL FIX
            if state_storage_address:
                with open('abis/StateStorage.json', 'r') as f:
                    state_storage_abi = json.load(f)
                
                self.state_storage = self.web3.eth.contract(
                    address=state_storage_address,
                    abi=state_storage_abi
                )
            else:
                # Try to get state storage address from Router if not provided
                try:
                    state_storage_address = self.router.functions.getImplementation(0).call()
                    with open('abis/StateStorage.json', 'r') as f:
                        state_storage_abi = json.load(f)
                    
                    self.state_storage = self.web3.eth.contract(
                        address=state_storage_address,
                        abi=state_storage_abi
                    )
                    logger.info(f"StateStorage contract initialized from Router: {state_storage_address}")
                except Exception as e:
                    logger.error(f"Failed to initialize StateStorage: {e}")
                    return False

            # Check if this account is authorized
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

    async def monitor_betting_rounds(self):
        """Monitor betting rounds and handle transitions"""
        while self.is_running:
            try:
                current_time = datetime.now()
                # Check betting rounds periodically
                if (current_time - self.last_round_check).total_seconds() >= self.ROUND_CHECK_INTERVAL:
                    self.last_round_check = current_time
                    
                    # Get current game state
                    game_state = await self.get_game_state()
                    if not game_state:
                        await asyncio.sleep(1)
                        continue
                    
                    # Store current round for comparison
                    current_round = game_state[2]  # currentRound from gameStateValues
                    
                    # Check if round is complete
                    if await self.is_betting_round_complete(game_state):
                        # If no active betting (current_turn is zero address)
                        if game_state[8] == "0x0000000000000000000000000000000000000000":
                            # Handle based on current round
                            if current_round == 0:  # PreFlop
                                await self.deal_flop()
                            elif current_round == 1:  # Flop
                                await self.deal_turn()
                            elif current_round == 2:  # Turn
                                await self.deal_river()
                
                await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error monitoring betting rounds: {e}")
                await asyncio.sleep(5)  # Longer sleep on error

    async def get_game_state(self):
        """Get current game state from StateStorage"""
        try:
            return self.state_storage.functions.getGameStateValues().call()
        except Exception as e:
            logger.error(f"Error getting game state: {e}")
            return None

    async def is_betting_round_complete(self, game_state):
        """Check if all players have acted and betting is equalized"""
        try:
            # If there's no current turn, it means the round is complete
            return game_state[8] == "0x0000000000000000000000000000000000000000"
        except Exception as e:
            logger.error(f"Error checking if round is complete: {e}")
            return False

    async def deal_flop(self):
        """Deal the flop through the HandManager contract"""
        try:
            logger.info("Dealing flop")
            
            # Get HandManager address from Router
            hand_manager_address = self.router.functions.getImplementation(3).call()
            
            # Load HandManager ABI
            with open('abis/HandManager.json', 'r') as f:
                hand_manager_abi = json.load(f)
            
            # Initialize HandManager contract
            hand_manager = self.web3.eth.contract(
                address=hand_manager_address,
                abi=hand_manager_abi
            )
            
            # Build transaction
            tx = hand_manager.functions.dealFlop().build_transaction({
                'from': self.account.address,
                'nonce': self.web3.eth.get_transaction_count(self.account.address),
                'gas': 300000,
                'gasPrice': int(self.web3.eth.gas_price * 1.1)
            })

            # Sign and send transaction with retry logic
            max_attempts = 3
            base_delay = 2  # seconds
            
            for attempt in range(max_attempts):
                try:
                    signed_tx = self.web3.eth.account.sign_transaction(tx, self.account.key)
                    tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                    receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                    
                    if receipt['status'] == 1:
                        logger.info(f"Flop dealt successfully (tx: {tx_hash.hex()})")
                        return True
                    else:
                        logger.error(f"Deal flop transaction failed (tx: {tx_hash.hex()})")
                        if attempt < max_attempts - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.info(f"Retrying in {delay}s (attempt {attempt+1}/{max_attempts})")
                            await asyncio.sleep(delay)
                        else:
                            return False
                except Exception as e:
                    logger.error(f"Error in transaction attempt {attempt+1}: {e}")
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.info(f"Retrying in {delay}s (attempt {attempt+1}/{max_attempts})")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Failed to deal flop after {max_attempts} attempts")
                        return False
            
            return False
        except Exception as e:
            logger.error(f"Error dealing flop: {e}")
            return False

    async def deal_turn(self):
        """Deal the turn through the HandManager contract"""
        try:
            logger.info("Dealing turn")
            
            # Get HandManager address from Router
            hand_manager_address = self.router.functions.getImplementation(3).call()
            
            # Load HandManager ABI
            with open('abis/HandManager.json', 'r') as f:
                hand_manager_abi = json.load(f)
            
            # Initialize HandManager contract
            hand_manager = self.web3.eth.contract(
                address=hand_manager_address,
                abi=hand_manager_abi
            )
            
            # Build transaction
            tx = hand_manager.functions.dealTurn().build_transaction({
                'from': self.account.address,
                'nonce': self.web3.eth.get_transaction_count(self.account.address),
                'gas': 300000,
                'gasPrice': int(self.web3.eth.gas_price * 1.1)
            })

            # Implementation with retry logic (similar to deal_flop)
            # Sign and send transaction with retry
            max_attempts = 3
            base_delay = 2
            
            for attempt in range(max_attempts):
                try:
                    signed_tx = self.web3.eth.account.sign_transaction(tx, self.account.key)
                    tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                    receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                    
                    if receipt['status'] == 1:
                        logger.info(f"Turn dealt successfully (tx: {tx_hash.hex()})")
                        return True
                    else:
                        logger.error(f"Deal turn transaction failed (tx: {tx_hash.hex()})")
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(base_delay * (2 ** attempt))
                        else:
                            return False
                except Exception as e:
                    logger.error(f"Error dealing turn (attempt {attempt+1}): {e}")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(base_delay * (2 ** attempt))
                    else:
                        return False
            
            return False

        except Exception as e:
            logger.error(f"Error dealing turn: {e}")
            return False

    async def deal_river(self):
        """Deal the river through the HandManager contract"""
        try:
            logger.info("Dealing river")
            
            # Get HandManager address from Router
            hand_manager_address = self.router.functions.getImplementation(3).call()
            
            # Load HandManager ABI
            with open('abis/HandManager.json', 'r') as f:
                hand_manager_abi = json.load(f)
            
            # Initialize HandManager contract
            hand_manager = self.web3.eth.contract(
                address=hand_manager_address,
                abi=hand_manager_abi
            )
            
            # Build transaction
            tx = hand_manager.functions.dealRiver().build_transaction({
                'from': self.account.address,
                'nonce': self.web3.eth.get_transaction_count(self.account.address),
                'gas': 300000,
                'gasPrice': int(self.web3.eth.gas_price * 1.1)
            })

            # Implementation with retry logic (similar to previous methods)
            max_attempts = 3
            base_delay = 2
            
            for attempt in range(max_attempts):
                try:
                    signed_tx = self.web3.eth.account.sign_transaction(tx, self.account.key)
                    tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                    receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                    
                    if receipt['status'] == 1:
                        logger.info(f"River dealt successfully (tx: {tx_hash.hex()})")
                        return True
                    else:
                        logger.error(f"Deal river transaction failed (tx: {tx_hash.hex()})")
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(base_delay * (2 ** attempt))
                        else:
                            return False
                except Exception as e:
                    logger.error(f"Error dealing river (attempt {attempt+1}): {e}")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(base_delay * (2 ** attempt))
                    else:
                        return False
            
            return False

        except Exception as e:
            logger.error(f"Error dealing river: {e}")
            return False

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
        """Monitor contract events for turns using getLogs with persistence"""
        # Start by getting the current block and initializing from a safe point
        if self.last_processed_block == 0:
            current_block = self.web3.eth.block_number
            self.last_processed_block = max(0, current_block - 1000)  # Safe initial value
        
        while self.is_running:
            try:
                # Get latest block number
                latest_block = self.web3.eth.block_number
                
                # Only proceed if there are new blocks to process
                if latest_block <= self.last_processed_block:
                    await asyncio.sleep(1)
                    continue
                    
                logger.info(f"Checking blocks {self.last_processed_block+1} to {latest_block}")
                
                # Get event signature (ensuring 0x prefix)
                event_signature = self.web3.keccak(
                    text="ActionTimerStarted(address,uint256,uint256)"
                ).hex()
                if not event_signature.startswith('0x'):
                    event_signature = '0x' + event_signature

                # Get logs
                logs = self.web3.eth.get_logs({
                    'address': self.game_logic.address,
                    'fromBlock': self.last_processed_block + 1,
                    'toBlock': latest_block,
                    'topics': [
                        event_signature
                        event_signature
                    ]
                })
                
                # Process logs
                for log in logs:
                    try:
                        # Generate unique event ID
                        event_id = f"{log['blockNumber']}-{log['transactionIndex']}-{log['logIndex']}"
                        
                        # Skip if we've already processed this event
                        if event_id in self.processed_events:
                            continue
                            
                        # Extract player address from the indexed parameter
                        player_address = self.web3.to_checksum_address('0x' + log['topics'][1].hex()[-40:])
                        
                        # Decode the non-indexed parameters from data
                        data = log['data']
                        decoded_data = self.web3.codec.decode_abi(['uint256', 'uint256'], bytes.fromhex(data[2:]))
                        duration = decoded_data[0]
                        
                        # Set expiry time
                        expiry_time = datetime.now() + timedelta(seconds=duration)
                        self.active_timers[player_address] = expiry_time
                        
                        # Mark this event as processed
                        self.processed_events.add(event_id)
                        
                        logger.info(f"Started timer for player {player_address}, expires at {expiry_time}")
                    except Exception as e:
                        logger.error(f"Error processing log: {e}")
                
                # Update the last processed block
                self.last_processed_block = latest_block
                
                # Keep the processed events set from growing too large
                if len(self.processed_events) > 10000:
                    self.processed_events = set(list(self.processed_events)[-5000:])

            except Exception as e:
                logger.error(f"Error monitoring events: {e}")
            
            await asyncio.sleep(2)  # Check every 2 seconds

    async def start(self):
        """Start the timer agent"""
        self.is_running = True
        logger.info("Starting timer agent...")
        
        try:
            # Run all monitoring methods concurrently
            await asyncio.gather(
                self.monitor_events(),
                self.monitor_timers(),
                self.monitor_blind_levels()
            )
        except Exception as e:
            logger.error(f"Error in timer agent main loop: {e}")
            self.is_running = False

    async def stop(self):
        """Stop the timer agent"""
        self.is_running = False
        logger.info("Stopping timer agent...")
        self.active_timers.clear()
