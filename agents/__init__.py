import sys
import os

# Force Python to look in the root directory of your project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Your existing imports continue below...
from agents.registry import AgentRegistry, AGENT_ICONS, AGENT_DESCRIPTIONS
