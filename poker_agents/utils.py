import logging
from typing import Optional, Dict
from ethers import JsonRpcProvider, Wallet, Contract
import json
import aiohttp
import json

logger = logging.getLogger(__name__)

async def get_openrouter_models():
    """Fetch available models from OpenRouter"""
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
        }
        async with session.get("https://openrouter.ai/api/v1/models", headers=headers) as response:
            if response.status == 200:
                models = await response.json()
                return models
            return None

async def validate_connection(rpc_url: str) -> bool:
    """Validate RPC connection"""
    try:
        provider = JsonRpcProvider(rpc_url)
        await provider.get_block_number()
        return True
    except Exception as e:
        logger.error(f"Connection validation failed: {e}")
        return False

async def check_wallet_balance(provider: JsonRpcProvider, address: str, 
                             min_balance: float) -> bool:
    """Check if wallet has sufficient balance"""
    try:
        balance = await provider.get_balance(address)
        return float(balance) >= min_balance
    except Exception as e:
        logger.error(f"Balance check failed: {e}")
        return False

def load_contract_abi(contract_name: str) -> Dict:
    """Load contract ABI from file"""
    try:
        with open(f'abis/{contract_name}.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load ABI for {contract_name}: {e}")
        raise

async def initialize_contract(
    address: str,
    abi: Dict,
    signer: Optional[Wallet] = None,
    provider: Optional[JsonRpcProvider] = None
) -> Contract:
    """Initialize a contract instance"""
    if not signer and not provider:
        raise ValueError("Either signer or provider must be provided")
    
    return Contract(
        address,
        abi,
        signer or provider
    )