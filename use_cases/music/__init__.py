from core.schema import UseCase
from .tools import TOOLS
from .prompt import MUSIC_SYSTEM_PROMPT

MusicUseCase = UseCase(
    display_name="Music Discovery",
    input_placeholder="e.g. I need something chill for studying",
    system_prompt=MUSIC_SYSTEM_PROMPT,
    tools=TOOLS,
)
