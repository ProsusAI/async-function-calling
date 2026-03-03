"""BenchScenario dataclass and the 6 single-message scenarios."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchScenario:
    name: str
    user_message: str
    success_marker: str   # substring expected in final assistant response
    description: str


SCENARIOS: dict[str, BenchScenario] = {
    "instant_only": BenchScenario(
        name="instant_only",
        user_message="What genres of music are best for a study session?",
        success_marker="Ambient",
        description=(
            "Control: only instant tools — sync and async should behave identically. "
            "Verifies the benchmark itself introduces no spurious differences."
        ),
    ),
    "single_slow": BenchScenario(
        name="single_slow",
        user_message="Recommend some jazz artists for me.",
        success_marker="Miles Davis",
        description=(
            "Single slow tool (search_artists, 2s). "
            "Measures the base overhead of async injection for one tool."
        ),
    ),
    "mixed_instant_slow": BenchScenario(
        name="mixed_instant_slow",
        user_message="Tell me about jazz as a genre and recommend some jazz artists.",
        success_marker="Miles Davis",
        description=(
            "One instant tool (get_genre_info) + one slow tool (search_artists). "
            "Tests correct mixed routing and incorporation of both results."
        ),
    ),
    "two_parallel": BenchScenario(
        name="two_parallel",
        user_message=(
            "Compare jazz and electronic music artists. "
            "Give me the best from each genre."
        ),
        success_marker="Miles Davis",
        description=(
            "Two parallel slow tools (both search_artists, ~2s each). "
            "Async runs both simultaneously — should be ~2× faster than sync."
        ),
    ),
    "three_parallel": BenchScenario(
        name="three_parallel",
        user_message=(
            "Find me jazz artists, build a chill study playlist, "
            "and search for ambient artists."
        ),
        success_marker="Miles Davis",
        description=(
            "Three parallel slow tools (search_artists ×2 + build_playlist). "
            "Maximum latency advantage; hardest case for async synthesis quality."
        ),
    ),
    "chain": BenchScenario(
        name="chain",
        user_message=(
            "Find me jazz artists. "
            "Then get the full discography for Miles Davis."
        ),
        success_marker="Kind of Blue",
        description=(
            "Dependent chain: search_artists → (await_job) → get_discography(Miles Davis). "
            "Tests await_job; async/tool should match sync quality when hint fires correctly."
        ),
    ),
}
