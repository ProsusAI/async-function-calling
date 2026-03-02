from core.schema import UseCase
from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS, SLOW_TOOLS
from .prompt import MUSIC_SYSTEM_PROMPT

MusicUseCase = UseCase(
    display_name="Music Discovery",
    input_placeholder="e.g. I need something chill for studying",
    system_prompt=MUSIC_SYSTEM_PROMPT,
    tool_schemas=TOOL_SCHEMAS,
    tool_functions=TOOL_FUNCTIONS,
    slow_tools=SLOW_TOOLS,
)
