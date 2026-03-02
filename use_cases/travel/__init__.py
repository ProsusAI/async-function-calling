from core.schema import UseCase
from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS, SLOW_TOOLS
from .prompt import TRAVEL_SYSTEM_PROMPT

TravelUseCase = UseCase(
    display_name="Travel Assistant",
    input_placeholder="e.g. Find flights from Tokyo to Amsterdam",
    system_prompt=TRAVEL_SYSTEM_PROMPT,
    tool_schemas=TOOL_SCHEMAS,
    tool_functions=TOOL_FUNCTIONS,
    slow_tools=SLOW_TOOLS,
)
