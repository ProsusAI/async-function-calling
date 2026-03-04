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
        # is_async=True  → agents fire in parallel background threads,
        #                   results injected when ready (fire-and-forget)
        # is_async=False → agents run sequentially, orchestrator blocks
        AgentTool(
            name="music_agent",
            description="Music specialist: recommendations, playlists, artists, genres, moods.",
            use_case=MusicUseCase,
            is_async=True,
        ),
        AgentTool(
            name="travel_agent",
            description="Travel specialist: flights, hotels, activities, destinations.",
            use_case=TravelUseCase,
            is_async=True,
        ),
    ],
)
