from core.schema import UseCase
from core.agent_tool import AgentTool
from use_cases.music import MusicUseCase
from use_cases.travel import TravelUseCase
from .prompt import MULTI_SYSTEM_PROMPT

MultiUseCase = UseCase(
    display_name="Multi-Agent Demo",
    input_placeholder="e.g. Plan a jazz-themed trip to Amsterdam",
    system_prompt=MULTI_SYSTEM_PROMPT,
    tools=[
        # is_async=True  → sub-agent fires in a parent background thread (parallel)
        # forced_sync=False → sub-agent runs its own async tools in parallel internally
        #
        # Flip forced_sync=True if you want the sub-agent to run its tools sequentially.
        # Flip is_async=False if you want the parent to block on each sub-agent in turn.
        AgentTool(
            name="music_agent",
            description="Music specialist: recommendations, playlists, artists, genres, moods.",
            use_case=MusicUseCase,
            is_async=True,
            forced_sync=False,
        ),
        AgentTool(
            name="travel_agent",
            description="Travel specialist: flights, hotels, activities, destinations.",
            use_case=TravelUseCase,
            is_async=True,
            forced_sync=False,
        ),
    ],
)
