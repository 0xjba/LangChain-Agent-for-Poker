import asyncio
import sys
from poker_agents.cli import main

if __name__ == "__main__":
    interactive = "--interactive" in sys.argv
    asyncio.run(main(interactive))