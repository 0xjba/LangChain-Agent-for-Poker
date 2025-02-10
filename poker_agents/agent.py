from typing import List, Dict, Optional
import logging
import asyncio
from dataclasses import dataclass
from web3 import Web3
import asyncio
import json
import logging
from typing import Optional, Set
from dataclasses import dataclass
from datetime import datetime
import json
import os
from datetime import datetime
from enum import IntEnum
from .openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

class BettingRound(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

class PlayerStatus(IntEnum):
    INACTIVE = 0
    ACTIVE = 1
    FOLDED = 2
    ELIMINATED = 3

@dataclass
class GameState:
    action_timer: int
    community_cards: List[int]
    current_round: BettingRound
    main_pot: int
    current_bet: int
    last_raise: int
    min_raise: int
    last_aggressor: int
    current_turn: str
    hand_start_time: int
    last_action_amount: int

@dataclass
class PlayerState:
    stack: int
    status: PlayerStatus
    current_bet: int
    position: int
    hole_cards: List[int]
    last_action_time: int

class PokerAgent:
    def __init__(self, model_name: str = "anthropic/claude-2"):
        self.is_running = False
        # Track turns we've already acted on to prevent duplicates
        self.processed_turns = set()
        self.last_action_time = None
        # Minimum time between actions (in seconds)
        self.MIN_ACTION_INTERVAL = 2
        
        # Initialize OpenRouter client
        self.llm = OpenRouterClient(model_name=model_name)
        
        # Define the system prompt directly
        self.system_prompt = (
            "You are an expert poker player making strategic decisions. "
            "Analyze the situation and choose the best action: FOLD (0), CHECK (1), CALL (2), or RAISE (3). "
            "If raising, also specify the raise amount.\n\n"
            "Consider:\n"
            "- Pot odds and implied odds\n"
            "- Position and table dynamics\n"
            "- Hand strength and potential\n"
            "- Stack sizes and tournament stage\n"
            "- Previous betting patterns\n"
            "- Tournament vs Cash game strategy\n\n"
            "Response format (JSON):\n"
            "{\n"
            '    "action": 0-3,\n'
            '    "amount": raise_amount (optional, only if action is 3),\n'
            '    "reasoning": "detailed explanation of the decision",\n'
            '    "confidence": 0-100\n'
            "}"
        )

    async def initialize(self, rpc_url: str, private_key: str, 
                        router_address: str, state_storage_address: str,
                        game_logic_address: str) -> bool:
        """Initialize the agent with Web3 and contract connections"""
        try:
            # Initialize Web3 and account
            self.web3 = Web3(Web3.HTTPProvider(rpc_url))
            self.account = self.web3.eth.account.from_key(private_key)
            
            # Load contract ABIs
            with open('abis/Router.json', 'r') as f:
                router_abi = json.load(f)
            with open('abis/StateStorage.json', 'r') as f:
                state_storage_abi = json.load(f)
            with open('abis/GameLogic.json', 'r') as f:
                game_logic_abi = json.load(f)

            # Initialize contracts
            self.router = self.web3.eth.contract(
                address=router_address,
                abi=router_abi
            )
            
            self.state_storage = self.web3.eth.contract(
                address=state_storage_address,
                abi=state_storage_abi
            )

            self.game_logic = self.web3.eth.contract(
                address=game_logic_address,
                abi=game_logic_abi
            )

            logger.info(f"Agent initialized with address: {self.account.address}")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def _load_abi(self, filename: str) -> dict:
        """Load contract ABI from file"""
        try:
            with open(f'abis/{filename}', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ABI {filename}: {e}")
            raise

    async def make_action(self, action_type: int, amount: int = 0) -> bool:
        """Execute poker action through Router contract"""
        try:
            # Prepare transaction data
            if action_type == 3:  # Raise
                data = self.web3.eth.abi.encode_abi(['uint256'], [amount])
            else:
                data = b''

            # Build transaction
            tx = self.router.functions.routeGameAction(
                action_type,
                data
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.web3.eth.get_transaction_count(self.account.address),
                'gas': 200000,  # Estimate this
                'gasPrice': self.web3.eth.gas_price
            })

            # Sign and send transaction
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction receipt
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt['status'] == 1:
                logger.info(f"Action completed: type={action_type}, amount={amount}")
                return True
            else:
                logger.error("Transaction failed")
                return False

        except Exception as e:
            logger.error(f"Error making action: {e}")
            return False

    async def get_game_state(self) -> GameState:
        """Get current game state from StateStorage"""
        try:
            state = self.state_storage.functions.getGameStateValues().call()
            return GameState(
                action_timer=state[0],
                community_cards=state[1],
                current_round=BettingRound(state[2]),
                main_pot=state[3],
                current_bet=state[4],
                last_raise=state[5],
                min_raise=state[6],
                last_aggressor=state[7],
                current_turn=state[8],
                hand_start_time=state[9],
                last_action_amount=state[10]
            )
        except Exception as e:
            logger.error(f"Error getting game state: {e}")
            raise

    async def get_player_state(self, address: str) -> PlayerState:
        """Get player state from StateStorage"""
        try:
            player = self.state_storage.functions.getPlayer(address).call()
            return PlayerState(
                stack=player[0],
                status=PlayerStatus(player[1]),
                current_bet=player[2],
                position=player[3],
                hole_cards=player[4],
                last_action_time=player[5]
            )
        except Exception as e:
            logger.error(f"Error getting player state: {e}")
            raise

    async def handle_turn(self, game_state: GameState, player_state: PlayerState):
        """Handle player's turn using LLM for decision making"""
        try:
            if self._is_recent_action():
                logger.debug("Skipping turn - too soon after last action")
                return

            # Get additional game information
            active_players = await self._count_active_players()
            previous_actions = await self._get_previous_actions()
            tournament_stage = await self._get_tournament_stage()
            
            # Format cards for readability
            hole_cards = self._format_cards(player_state.hole_cards)
            community_cards = self._format_cards(game_state.community_cards)

            # Format game state message
            game_state_message = (
                f"Game State:\n"
                f"Hand: {hole_cards}\n"
                f"Community Cards: {community_cards}\n"
                f"Current Bet: {game_state.current_bet}\n"
                f"Your Stack: {player_state.stack}\n"
                f"Pot Size: {game_state.main_pot}\n"
                f"Position: {player_state.position}\n"
                f"Active Players: {active_players}\n"
                f"Previous Actions: {previous_actions}\n"
                f"Current Round: {game_state.current_round}\n"
                f"Tournament Stage: {tournament_stage}"
            )

            # Create the messages array for OpenRouter
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": game_state_message
                }
            ]

            # Generate unique ID for this turn
            turn_id = self._generate_turn_id(game_state)
            
            if turn_id in self.processed_turns:
                logger.debug(f"Turn {turn_id} already processed")
                return

            # Get LLM decision
            response = self.llm.get_completion(messages)
            
            try:
                decision = json.loads(response)
                logger.info(f"Decision reasoning: {decision['reasoning']} (Confidence: {decision.get('confidence', 0)}%)")
                
                # Validate and execute decision
                action_type = decision['action']
                amount = decision.get('amount', 0)
                
                if not self._is_valid_action(action_type, amount, game_state, player_state):
                    logger.warning("LLM suggested invalid action, defaulting to FOLD")
                    await self.make_action(0)
                    return

                await self.make_action(action_type, amount)
                
                # Record this turn as processed
                self.processed_turns.add(turn_id)
                self.last_action_time = datetime.now()

                # Keep processed turns set from growing too large
                if len(self.processed_turns) > 1000:
                    self.processed_turns = set(list(self.processed_turns)[-500:])
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse LLM response: {e}")
                await self.make_action(0)

        except Exception as e:
            logger.error(f"Error in turn handling: {e}")
            await self.make_action(0)

    def _format_cards(self, cards: List[int]) -> str:
        """Format cards into readable strings"""
        suits = ['♥', '♦', '♣', '♠']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        
        formatted = []
        for card in cards:
            if card == 0:  # Skip empty/hidden cards
                continue
            suit = suits[card // 13]
            rank = ranks[card % 13]
            formatted.append(f"{rank}{suit}")
        
        return ' '.join(formatted)

    async def _count_active_players(self) -> int:
        """Count number of active players"""
        tournament_state = await self.state_storage.getTournamentStateValues()
        return tournament_state[7]  # activePlayerCount

    async def _get_previous_actions(self) -> List[str]:
        """Get previous actions in current round"""
        # This would need to be implemented based on your contract's event system
        # Placeholder implementation
        return []

    async def _get_tournament_stage(self) -> str:
        """Get current tournament stage information"""
        tournament_state = await self.state_storage.getTournamentStateValues()
        return f"Level {tournament_state[10]}, Blinds: {tournament_state[0]}/{tournament_state[1]}"

    def _is_valid_action(self, action_type: int, amount: int, 
                        game_state: GameState, player_state: PlayerState) -> bool:
        """Validate if an action is possible"""
        try:
            if action_type not in [0, 1, 2, 3]:
                return False
                
            if action_type == 1:  # CHECK
                return game_state.current_bet == 0 or player_state.current_bet == game_state.current_bet
                
            if action_type == 2:  # CALL
                call_amount = game_state.current_bet - player_state.current_bet
                return player_state.stack >= call_amount
                
            if action_type == 3:  # RAISE
                total_required = game_state.current_bet - player_state.current_bet + amount
                min_raise = game_state.current_bet * 2
                return (player_state.stack >= total_required and 
                       amount >= min_raise)
                
            return True  # FOLD is always valid
            
        except Exception as e:
            logger.error(f"Error in action validation: {e}")
            return False


    async def monitor_game_state(self):
        """Monitor game state continuously"""
        while self.is_running:
            try:
                game_state = await self.get_game_state()
                player_state = await self.get_player_state(self.account.address)

                # Check if it's our turn
                if game_state.current_turn == self.account.address:
                    await self.handle_turn(game_state, player_state)

                # Check if player is eliminated
                if player_state.status == PlayerStatus.ELIMINATED:
                    logger.info("Player eliminated - stopping agent")
                    self.is_running = False
                    break

                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in game state monitoring: {e}")
                await asyncio.sleep(1)

    def _is_recent_action(self) -> bool:
        """Check if we've acted recently to prevent duplicate actions"""
        if not self.last_action_time:
            return False
        elapsed = (datetime.now() - self.last_action_time).total_seconds()
        return elapsed < self.MIN_ACTION_INTERVAL
    
    def _generate_turn_id(self, game_state: GameState) -> str:
        """Generate unique ID for this turn to prevent duplicate actions"""
        return f"{game_state.hand_start_time}_{game_state.current_round}_{game_state.current_bet}"

    async def process_turn(self, game_state: Optional[GameState] = None):
        """Process a turn, with duplicate action prevention"""
        try:
            if self._is_recent_action():
                logger.debug("Skipping turn - too soon after last action")
                return

            if not game_state:
                game_state = await self.get_game_state()

            # Generate unique ID for this turn
            turn_id = self._generate_turn_id(game_state)
            
            if turn_id in self.processed_turns:
                logger.debug(f"Turn {turn_id} already processed")
                return

            # Verify it's still our turn
            if game_state.current_turn != self.account.address:
                return

            player_state = await self.get_player_state(self.account.address)
            await self.handle_turn(game_state, player_state)
            
            # Record this turn as processed
            self.processed_turns.add(turn_id)
            self.last_action_time = datetime.now()

            # Keep processed turns set from growing too large
            if len(self.processed_turns) > 1000:
                self.processed_turns = set(list(self.processed_turns)[-500:])

        except Exception as e:
            logger.error(f"Error processing turn: {e}")

    async def monitor_events(self):
        """Monitor contract events for turns using getLogs"""
        while self.is_running:
            try:
                # Get latest block number
                latest_block = self.web3.eth.block_number
                from_block = max(0, latest_block - 10)  # Look back 10 blocks

                # Event signature for ActionTimerStarted(address,uint256,uint256)
                event_signature = '0x' + self.web3.keccak(
                    text="ActionTimerStarted(address,uint256,uint256)"
                ).hex()
                event_signature = self.web3.to_hex(hexstr=event_signature)

                # Format player address as topic
                player_topic = self.web3.to_hex(
                    hexstr=self.account.address[2:].zfill(64)
                )
                
                # Get logs
                logs = self.web3.eth.get_logs({
                    'address': self.game_logic.address,
                    'fromBlock': from_block,
                    'toBlock': 'latest',
                    'topics': [
                        event_signature,
                        "0x" + self.account.address[2:].zfill(64)  # Make sure this has 0x prefix too
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
                logger.error(f"Error in event monitoring: {e}", exc_info=True)
            
            await asyncio.sleep(1)

    async def monitor_game_state(self):
        """Monitor game state through polling"""
        while self.is_running:
            try:
                game_state = await self.get_game_state()
                player_state = await self.get_player_state(self.account.address)

                # Check if it's our turn
                if game_state.current_turn == self.account.address:
                    await self.process_turn(game_state)

                # Check if player is eliminated
                if player_state.status == PlayerStatus.ELIMINATED:
                    logger.info("Player eliminated - stopping agent")
                    self.is_running = False
                    break

                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in game state monitoring: {e}")
                await asyncio.sleep(1)        

    async def start(self):
        """Start the agent with both monitoring methods"""
        self.is_running = True
        logger.info(f"Starting poker agent with address {self.account.address}")
        logger.info(f"Monitoring GameLogic contract at {self.game_logic.address}")
        
        try:
            # Run both monitoring methods concurrently
            await asyncio.gather(
                self.monitor_game_state(),
                self.monitor_events()
            )
        except Exception as e:
            logger.error(f"Error in agent main loop: {e}")
            self.is_running = False

    async def stop(self):
        """Stop the agent"""
        self.is_running = False
        logger.info("Stopping poker agent...")