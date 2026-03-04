from core.schema import UseCase
from .tools import TOOLS
from .prompt import TRAVEL_SYSTEM_PROMPT

TravelUseCase = UseCase(
    display_name="Travel Assistant",
    input_placeholder="e.g. Find flights from Tokyo to Amsterdam",
    system_prompt=TRAVEL_SYSTEM_PROMPT,
    tools=TOOLS,
)
