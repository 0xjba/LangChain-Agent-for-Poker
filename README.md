# Poker Agents

An autonomous poker agent system powered by OpenRouter LLM that interacts with smart contracts to play poker games. The agent uses advanced AI decision-making to play poker strategically while handling all blockchain interactions automatically.

## Features

- AI-powered poker decision making using OpenRouter (Claude-2)
- Run multiple autonomous poker agents simultaneously
- Interactive CLI interface for agent management
- Session-based contract management
- Secure private key handling
- Real-time game state monitoring
- Profile-based agent configuration
- Detailed strategy reasoning and logging

## Prerequisites

- Python 3.8+
- OpenRouter API key
- Ethereum RPC endpoint
- Smart contract addresses (Router and StateStorage)
- Wallet with sufficient balance for transactions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/0xjba/LangChain-Agent-for-Poker.git
cd poker-agents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENROUTER_API_KEY="your-api-key"
export OPENROUTER_REFERRER="your-site-url"  # Optional
```

4. Add contract ABIs:
Place your contract ABIs in the `abis` directory:
- `abis/Router.json`
- `abis/StateStorage.json`
- - `abis/GameLogic.json`

## Usage

### Using the CLI Interface

1. Start the CLI:
```bash
python run.py
```

2. First-time setup:
```
a. Configure OpenRouter:
   - Enter API key (if not set in environment)
   - Select LLM model
   - Set temperature for decision making

b. Set contract addresses:
   - Router contract address
   - StateStorage contract address
   - GameLogic contract address

c. Create agent profiles:
   - Profile name
   - RPC URL
   - Private key
```

3. Managing Agents:
```
Main Menu Options:
1. Manage Profiles - Create, view, edit, or delete agent profiles
2. Start Agents - Start single or multiple agents
3. Stop Agents - Stop running agents
4. View Status - Monitor agent status and game progress
5. Update Contract Addresses - Change contract addresses for the session
6. Exit - Close the application
```

### Using the Code Directly

You can integrate the PokerAgent class into your own code:

```python
from poker_agents.agent import PokerAgent
import asyncio
import os

async def run_agent():
    # Ensure OpenRouter API key is set
    os.environ["OPENROUTER_API_KEY"] = "your-api-key"
    
    agent = PokerAgent()
    
    # Initialize agent
    success = await agent.initialize(
        rpc_url="YOUR_RPC_URL",
        private_key="YOUR_PRIVATE_KEY",
        router_address="ROUTER_CONTRACT_ADDRESS",
        state_storage_address="STATE_STORAGE_ADDRESS"
    )
    
    if success:
        # Start the agent
        await agent.start()
    else:
        print("Failed to initialize agent")

# Run the agent
asyncio.run(run_agent())
```

### Running Multiple Agents

Using CLI:
```bash
1. Create multiple profiles with different wallets
2. Select "Start Multiple Agents" from the menu
3. Choose which agents to start
```

Using Code:
```python
async def run_multiple_agents():
    # Create and configure agents
    agent1 = PokerAgent()
    agent2 = PokerAgent()
    
    # Initialize with different wallets
    await agent1.initialize(rpc_url="URL1", private_key="KEY1", ...)
    await agent2.initialize(rpc_url="URL2", private_key="KEY2", ...)
    
    # Run agents concurrently
    await asyncio.gather(
        agent1.start(),
        agent2.start()
    )
```

## AI Strategy

The agent uses OpenRouter's Claude-2 model to make poker decisions based on:
- Pot odds and implied odds
- Position and table dynamics
- Hand strength and potential
- Stack sizes and tournament stage
- Previous betting patterns
- Tournament vs Cash game considerations

Each decision includes:
- Action selection (Fold, Check, Call, or Raise)
- Raise amount calculation (when applicable)
- Detailed reasoning
- Confidence level

## Configuration

### Profile Structure
```json
{
    "profiles": {
        "agent1": {
            "id": "unique_id",
            "rpc_url": "your_rpc_url",
            "private_key": "encrypted_key",
            "created_at": "timestamp",
            "last_used": "timestamp"
        }
    }
}
```

### Available LLM Models
- anthropic/claude-2 (default)
- anthropic/claude-instant-v1
- google/palm-2-chat-bison
- meta-llama/llama-2-70b-chat
- mistral/mistral-7b-instruct
& more

## Timer Agent

The Timer Agent is a crucial component that works alongside the poker AI agents to manage game timeouts. It acts as an authorized backend service that monitors players' turn timers and enforces timeouts when players exceed their allotted time.

### How it Works

1. Monitors `ActionTimerStarted` events from the game contract
2. Tracks active timers for each player
3. Automatically calls `routeTimeoutAction` when a player's time expires
4. Must use a wallet that's authorized as a timer backend

### Setup Steps

1. First-time setup:
   ```bash
   # Ensure your timer wallet is authorized by the contract admin
   export TIMER_PRIVATE_KEY="your-authorized-wallet-private-key"
   ```

2. In the CLI:
   ```
   1. Set up contracts (Router, StateStorage, GameLogic addresses)
   2. Start Timer Agent first (Main Menu -> Manage Timer Agent)
   3. Then start your AI agents
   ```

### Order of Operations

1. Initialize contract addresses
2. Start Timer Agent (required for game timing)
3. Create AI agent profiles
4. Start AI agents
5. Monitor both Timer and AI agents through CLI

### Important Notes

- The Timer Agent must run continuously during gameplay
- Uses a separate wallet from your AI agents
- Wallet must be pre-authorized as a timer backend
- One Timer Agent can handle multiple AI agents
- Automatically cleans up when stopping agents

### CLI Management

```
Manage Timer Agent Menu:
1. Start Timer Agent - Initialize and start the timer service
2. Stop Timer Agent - Stop the timer service
3. View Timer Status - Check active timers and agent status
4. Back to Main Menu
```

Monitor active timers and their status through the CLI interface to ensure proper game timing management.

## Monitoring

1. Real-time status through CLI:
```
- Running agents
- Current game round
- AI decisions and reasoning
- Transaction status
```

2. Logging:
```
- Detailed decision logs
- Strategy reasoning
- Transaction details
- Error tracking
```

## Safety Features

1. Private Key Security:
   - Encrypted storage
   - Memory-only usage
   - Never logged or displayed

2. Transaction Safety:
   - Action validation
   - Balance checking
   - Gas estimation
   - Error recovery

3. AI Decision Validation:
   - Action feasibility check
   - Stack size verification
   - Bet amount validation

## Error Recovery

The agent includes multiple safety mechanisms:
- Network disconnection handling
- Transaction failure recovery
- Invalid action prevention
- Automatic state recovery
- Graceful shutdown
