"""
Music Discovery use case — tool implementations.

Async tools (fire-and-forget, background threads): search_artists, get_discography, build_playlist
Sync tools  (inline, instant):                     get_genre_info, get_mood_genres
"""

import time

from core.schema import Tool

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_ARTISTS = {
    "jazz": [
        {"name": "Miles Davis",       "origin": "USA",     "era": "1950–1991", "style": "cool jazz, modal jazz"},
        {"name": "John Coltrane",     "origin": "USA",     "era": "1955–1967", "style": "hard bop, free jazz"},
        {"name": "Herbie Hancock",    "origin": "USA",     "era": "1962–present", "style": "post-bop, jazz-funk"},
        {"name": "Kamasi Washington", "origin": "USA",     "era": "2010–present", "style": "spiritual jazz, neo-soul"},
        {"name": "Norah Jones",       "origin": "USA",     "era": "2001–present", "style": "contemporary jazz, pop"},
    ],
    "electronic": [
        {"name": "Aphex Twin",   "origin": "UK",     "era": "1991–present", "style": "IDM, ambient techno"},
        {"name": "Daft Punk",    "origin": "France", "era": "1993–2021",    "style": "house, electronic pop"},
        {"name": "Brian Eno",    "origin": "UK",     "era": "1970–present", "style": "ambient, experimental"},
        {"name": "Four Tet",     "origin": "UK",     "era": "2001–present", "style": "folktronica, microhouse"},
        {"name": "Burial",       "origin": "UK",     "era": "2005–present", "style": "dubstep, UK garage"},
    ],
    "classical": [
        {"name": "Johann Sebastian Bach",  "origin": "Germany", "era": "1685–1750", "style": "Baroque"},
        {"name": "Frédéric Chopin",        "origin": "Poland",  "era": "1810–1849", "style": "Romantic"},
        {"name": "Ludovico Einaudi",       "origin": "Italy",   "era": "1988–present", "style": "contemporary classical"},
        {"name": "Max Richter",            "origin": "Germany", "era": "2000–present", "style": "post-minimal, neo-classical"},
        {"name": "Ólafur Arnalds",         "origin": "Iceland", "era": "2007–present", "style": "neo-classical, ambient"},
    ],
    "indie": [
        {"name": "Bon Iver",         "origin": "USA", "era": "2007–present", "style": "indie folk, art pop"},
        {"name": "Radiohead",        "origin": "UK",  "era": "1985–present", "style": "alternative rock, art rock"},
        {"name": "Phoebe Bridgers",  "origin": "USA", "era": "2017–present", "style": "indie folk, emo"},
        {"name": "The National",     "origin": "USA", "era": "1999–present", "style": "indie rock, baroque pop"},
        {"name": "Sufjan Stevens",   "origin": "USA", "era": "1999–present", "style": "indie folk, chamber pop"},
    ],
    "hip-hop": [
        {"name": "Kendrick Lamar",     "origin": "USA", "era": "2003–present", "style": "conscious hip-hop, West Coast"},
        {"name": "J. Cole",            "origin": "USA", "era": "2007–present", "style": "conscious hip-hop, Southern"},
        {"name": "Frank Ocean",        "origin": "USA", "era": "2011–present", "style": "R&B, neo-soul, hip-hop"},
        {"name": "Tyler, the Creator", "origin": "USA", "era": "2009–present", "style": "alternative hip-hop"},
        {"name": "Loyle Carner",       "origin": "UK",  "era": "2014–present", "style": "UK hip-hop, neo-soul"},
    ],
    "ambient": [
        {"name": "Brian Eno",            "origin": "UK",      "era": "1970–present", "style": "ambient, experimental"},
        {"name": "Stars of the Lid",     "origin": "USA",     "era": "1993–present", "style": "drone, symphonic ambient"},
        {"name": "William Basinski",     "origin": "USA",     "era": "1982–present", "style": "decay loops, tape music"},
        {"name": "Hammock",              "origin": "USA",     "era": "2003–present", "style": "ambient, post-rock"},
        {"name": "Ólafur Arnalds",       "origin": "Iceland", "era": "2007–present", "style": "neo-classical, ambient"},
    ],
}

_DISCOGRAPHY = {
    "Miles Davis": {
        "albums": [
            "Kind of Blue (1959) — 5★  modal masterpiece, best-selling jazz album of all time",
            "Bitches Brew (1970) — 5★  electric jazz-fusion landmark",
            "In a Silent Way (1969) — 4★  hypnotic fusion, proto-ambient",
            "Round About Midnight (1957) — 4★  bop standards, debut on Columbia",
        ],
        "top_tracks": ["So What", "All Blues", "Freddie Freeloader", "Blue in Green", "Flamenco Sketches", "Birdland"],
    },
    "Aphex Twin": {
        "albums": [
            "Selected Ambient Works 85–92 (1992) — 5★  pioneering IDM & ambient techno",
            "Richard D. James Album (1996) — 5★  breakcore meets melody",
            "Drukqs (2001) — 4★  prepared piano and hyperactive breakbeats",
            "Syro (2014) — 4★  polished comeback, Warp Records",
        ],
        "top_tracks": ["Xtal", "Windowlicker", "Come to Daddy", "Alberto Balsalm", "Avril 14th", "Ageispolis"],
    },
    "Bon Iver": {
        "albums": [
            "For Emma, Forever Ago (2007) — 5★  recorded alone in a Wisconsin cabin",
            "Bon Iver, Bon Iver (2011) — 5★  lush, orchestral, Grammy-winning",
            "22, A Million (2016) — 4★  deconstructed folk, heavy processing",
            "i,i (2019) — 4★  warm resolution of the trilogy",
        ],
        "top_tracks": ["Skinny Love", "Holocene", "Flume", "Towers", "Perth", "Hey, Ma"],
    },
    "Kendrick Lamar": {
        "albums": [
            "good kid, m.A.A.d city (2012) — 5★  Compton coming-of-age short film in music",
            "To Pimp a Butterfly (2015) — 5★  jazz-funk protest album, Pulitzer Prize",
            "DAMN. (2017) — 5★  pop crossover, Grammy Album of the Year",
            "Mr. Morale & the Big Steppers (2022) — 4★  therapy sessions as hip-hop",
        ],
        "top_tracks": ["HUMBLE.", "Alright", "Money Trees", "Swimming Pools", "King Kunta", "DNA."],
    },
    "Ludovico Einaudi": {
        "albums": [
            "Divenire (2006) — 5★  orchestral minimalism, breakout record",
            "Nightbook (2009) — 4★  darker, more introspective",
            "In a Time Lapse (2013) — 5★  broad strokes, cinematic scope",
            "Seven Days Walking (2019) — 4★  seven albums exploring one theme",
        ],
        "top_tracks": ["Experience", "Nuvole Bianche", "Una Mattina", "Divenire", "Life", "Primavera"],
    },
    "Brian Eno": {
        "albums": [
            "Ambient 1: Music for Airports (1978) — 5★  invented the ambient genre",
            "Another Green World (1975) — 5★  art-rock meets tape experiments",
            "Discreet Music (1975) — 4★  generative systems, tape loops",
            "Apollo: Atmospheres & Soundtracks (1983) — 4★  cosmic synth, for NASA film",
        ],
        "top_tracks": ["1/1", "By This River", "An Ending (Ascent)", "Sombre Reptiles", "Spirits Drifting"],
    },
    "Radiohead": {
        "albums": [
            "OK Computer (1997) — 5★  alienation, machines, and modern anxiety",
            "Kid A (2000) — 5★  post-rock, electronic, abandoned rock entirely",
            "In Rainbows (2007) — 5★  warm, emotional, pay-what-you-want release",
            "The Bends (1995) — 4★  alt-rock touchstone",
        ],
        "top_tracks": ["Karma Police", "Fake Plastic Trees", "Paranoid Android", "No Surprises", "Creep", "Nude"],
    },
    "Phoebe Bridgers": {
        "albums": [
            "Stranger in the Alps (2017) — 4★  debut: ghosts, grief, and guitar",
            "Punisher (2020) — 5★  quarantine-era masterpiece, indie folk landmark",
        ],
        "top_tracks": ["Motion Sickness", "Savior Complex", "Garden Song", "Moon Song", "Funeral", "Kyoto"],
    },
}

_GENRE_INFO = {
    "jazz": {
        "bpm_range": "60–180",
        "characteristics": "improvisation, swing rhythm, blues harmony, walking bass, complex chord changes",
        "sub_genres": ["bebop", "cool jazz", "free jazz", "fusion", "modal jazz", "smooth jazz"],
        "listen_for": "call-and-response between instruments, spontaneous melody invention",
    },
    "electronic": {
        "bpm_range": "90–175",
        "characteristics": "synthesizers, drum machines, sequencers, sampling, sound design",
        "sub_genres": ["techno", "house", "IDM", "ambient", "trance", "drum & bass"],
        "listen_for": "texture layers, rhythmic patterns, filter sweeps, sound evolution over time",
    },
    "classical": {
        "bpm_range": "40–200",
        "characteristics": "orchestral instruments, formal structure, written notation, dynamic contrast",
        "sub_genres": ["Baroque", "Classical", "Romantic", "Impressionist", "Contemporary", "Minimalist"],
        "listen_for": "thematic development, counterpoint, orchestration choices",
    },
    "indie": {
        "bpm_range": "70–145",
        "characteristics": "independent production, DIY aesthetic, introspective lyrics, raw sound",
        "sub_genres": ["indie rock", "indie folk", "indie pop", "dream pop", "shoegaze", "lo-fi"],
        "listen_for": "lyrical detail, unconventional song structures, bedroom production warmth",
    },
    "hip-hop": {
        "bpm_range": "70–100",
        "characteristics": "rhythmic vocals, sampling, drum programming, storytelling, flow",
        "sub_genres": ["conscious", "trap", "lo-fi", "boom bap", "alternative", "cloud rap"],
        "listen_for": "internal rhyme schemes, sample flips, producer signatures",
    },
    "ambient": {
        "bpm_range": "40–90",
        "characteristics": "texture over melody, slow evolution, atmospheric, minimal, often instrumental",
        "sub_genres": ["dark ambient", "drone", "space music", "neo-classical", "new age", "field recordings"],
        "listen_for": "gradual harmonic shifts, space between notes, how silence is used",
    },
}

_MOOD_GENRES = {
    "study":    {"best": ["ambient", "classical", "jazz"],        "why": "minimal distractions, steady tempo, no disruptive dynamics"},
    "workout":  {"best": ["electronic", "hip-hop"],               "why": "driving BPM, high energy, motivational lyrics"},
    "chill":    {"best": ["jazz", "indie", "ambient"],            "why": "relaxed tempo, warm timbres, gentle progressions"},
    "party":    {"best": ["electronic", "hip-hop"],               "why": "dance-floor BPM, crowd energy, bass-heavy"},
    "sad":      {"best": ["indie", "classical", "ambient"],       "why": "emotional resonance, space for reflection"},
    "energetic":{"best": ["electronic", "hip-hop", "indie"],      "why": "fast tempo, bright tones, forward momentum"},
    "focus":    {"best": ["ambient", "classical", "electronic"],  "why": "no lyrics to parse, consistent texture, non-intrusive"},
    "creative": {"best": ["jazz", "electronic", "indie"],         "why": "harmonic surprise, genre-blending, unpredictable structure"},
    "sleep":    {"best": ["ambient", "classical"],                "why": "slow evolution, low dynamics, no sudden changes"},
}

_PLAYLIST_TRACKS = [
    ("So What",           "Miles Davis"),
    ("Xtal",              "Aphex Twin"),
    ("Skinny Love",       "Bon Iver"),
    ("HUMBLE.",           "Kendrick Lamar"),
    ("Experience",        "Ludovico Einaudi"),
    ("1/1",               "Brian Eno"),
    ("Karma Police",      "Radiohead"),
    ("Motion Sickness",   "Phoebe Bridgers"),
    ("Blue in Green",     "Miles Davis"),
    ("Windowlicker",      "Aphex Twin"),
    ("Holocene",          "Bon Iver"),
    ("Alright",           "Kendrick Lamar"),
    ("Nuvole Bianche",    "Ludovico Einaudi"),
    ("By This River",     "Brian Eno"),
    ("No Surprises",      "Radiohead"),
    ("Garden Song",       "Phoebe Bridgers"),
    ("All Blues",         "Miles Davis"),
    ("Alberto Balsalm",   "Aphex Twin"),
    ("Flume",             "Bon Iver"),
    ("Money Trees",       "Kendrick Lamar"),
]

# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def search_artists(args: dict) -> str:
    genre = args.get("genre", "").lower().strip()
    mood  = args.get("mood",  "").lower().strip()

    time.sleep(5)  # simulated catalog search

    # Resolve mood → genre when no genre given
    if mood and not genre:
        genre = _MOOD_GENRES.get(mood, {}).get("best", ["indie"])[0]

    artists = _ARTISTS.get(genre)
    if not artists:
        available = ", ".join(_ARTISTS)
        return f"Genre '{genre}' not found. Available: {available}"

    header = f"Artists — {genre.title()}"
    if mood:
        header += f" (matched from mood: {mood})"
    lines = [header, ""]
    for i, a in enumerate(artists, 1):
        lines.append(f"{i}. {a['name']}  [{a['origin']}, {a['era']}]")
        lines.append(f"   Style: {a['style']}")
    return "\n".join(lines)


def get_discography(args: dict) -> str:
    artist = args.get("artist", "").strip()

    time.sleep(8)  # simulated discography fetch

    data = _DISCOGRAPHY.get(artist)
    if not data:
        available = ", ".join(_DISCOGRAPHY)
        return f"'{artist}' not in catalog. Known artists: {available}"

    lines = [f"Discography — {artist}", ""]
    lines.append("Albums:")
    for album in data["albums"]:
        lines.append(f"  • {album}")
    lines.append("")
    lines.append("Top tracks:")
    for track in data["top_tracks"]:
        lines.append(f"  ♪  {track}")
    return "\n".join(lines)


def build_playlist(args: dict) -> str:
    theme  = args.get("theme",  "mixed").strip()
    vibe   = args.get("vibe",   "chill").strip()
    length = min(int(args.get("length", 10)), 20)

    time.sleep(12)  # playlist curation is the slowest operation

    tracks = _PLAYLIST_TRACKS[:length]

    lines = [f'Playlist: "{theme}" · {vibe}  ({length} tracks)', "─" * 42]
    for i, (track, artist) in enumerate(tracks, 1):
        lines.append(f"  {i:>2}.  {track:<28} {artist}")
    duration = length * 4
    lines.append("─" * 42)
    lines.append(f"  {length} tracks · ~{duration} min")
    return "\n".join(lines)


def get_genre_info(args: dict) -> str:
    genre = args.get("genre", "").lower().strip()
    info  = _GENRE_INFO.get(genre)
    if not info:
        available = ", ".join(_GENRE_INFO)
        return f"Genre '{genre}' not recognized. Available: {available}"

    return (
        f"{genre.title()} — quick facts\n"
        f"  BPM range:       {info['bpm_range']}\n"
        f"  Characteristics: {info['characteristics']}\n"
        f"  Sub-genres:      {', '.join(info['sub_genres'])}\n"
        f"  Listen for:      {info['listen_for']}"
    )


def get_mood_genres(args: dict) -> str:
    mood = args.get("mood", "").lower().strip()
    data = _MOOD_GENRES.get(mood)
    if not data:
        available = ", ".join(_MOOD_GENRES)
        return f"Mood '{mood}' not recognized. Try: {available}"

    genres = ", ".join(g.title() for g in data["best"])
    return f"Best genres for '{mood}': {genres}\nWhy: {data['why']}"


# ---------------------------------------------------------------------------
# Tool registry — sync/async declared per-tool, not in a separate set
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="search_artists",
        description=(
            "Search for artists by genre and/or listening mood. "
            "Returns a ranked list with style and era info. "
            "Async — dispatched in background. Provide genre, mood, or both."
        ),
        parameters={
            "type": "object",
            "properties": {
                "genre": {
                    "type": "string",
                    "description": "Music genre: jazz, electronic, classical, indie, hip-hop, ambient",
                },
                "mood": {
                    "type": "string",
                    "description": "Listening mood: study, workout, chill, party, sad, energetic, focus, creative, sleep",
                },
            },
        },
        fn=search_artists,
        is_async=True,
    ),
    Tool(
        name="get_discography",
        description=(
            "Fetch an artist's full album catalog and top tracks with ratings. "
            "Async — use await_job if you plan to recommend specific records right after."
        ),
        parameters={
            "type": "object",
            "properties": {
                "artist": {
                    "type": "string",
                    "description": "Artist name exactly as returned by search_artists, e.g. 'Miles Davis'",
                },
            },
            "required": ["artist"],
        },
        fn=get_discography,
        is_async=True,
    ),
    Tool(
        name="build_playlist",
        description=(
            "Curate a playlist for a theme and vibe. Returns an ordered tracklist. "
            "Async — use after you know what genre/mood the user wants."
        ),
        parameters={
            "type": "object",
            "properties": {
                "theme": {
                    "type": "string",
                    "description": "Playlist theme, e.g. 'late night jazz', 'rainy Sunday', 'morning run'",
                },
                "vibe": {
                    "type": "string",
                    "description": "Emotional tone: chill, energetic, melancholic, uplifting, focused, dreamy",
                },
                "length": {
                    "type": "integer",
                    "description": "Number of tracks (default 10, max 20)",
                },
            },
            "required": ["theme", "vibe"],
        },
        fn=build_playlist,
        is_async=True,
    ),
    Tool(
        name="get_genre_info",
        description=(
            "Instantly returns BPM range, defining characteristics, sub-genres, and listening tips "
            "for a music genre. Use before or alongside search_artists to give context."
        ),
        parameters={
            "type": "object",
            "properties": {
                "genre": {
                    "type": "string",
                    "description": "Genre name: jazz, electronic, classical, indie, hip-hop, ambient",
                },
            },
            "required": ["genre"],
        },
        fn=get_genre_info,
        is_async=False,
    ),
    Tool(
        name="get_mood_genres",
        description=(
            "Instantly maps a listening mood to the best matching genres with reasoning. "
            "Use this first when the user describes a mood rather than a genre."
        ),
        parameters={
            "type": "object",
            "properties": {
                "mood": {
                    "type": "string",
                    "description": "Listening mood: study, workout, chill, party, sad, energetic, focus, creative, sleep",
                },
            },
            "required": ["mood"],
        },
        fn=get_mood_genres,
        is_async=False,
    ),
]
