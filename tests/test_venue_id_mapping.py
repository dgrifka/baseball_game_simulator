"""
Regression tests for the UNIQLO Field / Dodger Stadium venue mapping bug.

The 2026 MLB v1 schedule API returned some Dodgers home games as
venue.name="UNIQLO Field at Dodger Stadium" (mixed case) with the correct
venue.id=22 (real Dodger Stadium). The live ingestion path was case-sensitive
on venue NAME instead of keying off the stable numeric venue.id:

  - Path A (LIVE): Simulator.get_game_information.fetch_games re-derived
    venue.id from venue.name through an ALL-CAPS-only mapping, degrading a
    mixed-case name to the 'neutral' venue id (wrong park factor + missing
    STADIUM_DIMENSIONS geometry).

(A second path, Simulator.best_batted_balls.fetch_games_by_date_range, had the
same class of bug but is dead code — imported by nothing and independently
FileNotFoundError-broken at import on main since its model file was deleted —
so it is out of scope here and left untouched.)

Path A must key off numeric venue.id and canonicalize venue.name FROM the id,
so a UNIQLO row always ends up venue.id == '22',
venue.name == 'Dodger Stadium' (the exact STADIUM_DIMENSIONS geometry key).
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytz

from Simulator.constants import (
    DEFAULT_VENUE_ID,
    NEUTRAL_SITE_VENUES,
    VALID_VENUE_IDS,
    VENUE_ID_TO_NAME,
    VENUE_NAME_TO_ID,
)


def _make_game(game_pk, venue_name, venue_id, game_dt,
                home="Los Angeles Dodgers", away="San Diego Padres",
                home_id=119, away_id=135):
    """Build a fabricated MLB Stats API schedule 'game' record."""
    return {
        "gamePk": game_pk,
        "officialDate": game_dt.strftime("%Y-%m-%d"),
        "gameDate": game_dt.isoformat(),
        "status": {"abstractGameState": "Final"},
        "venue": {"name": venue_name, "id": venue_id},
        "teams": {
            "away": {"team": {"id": away_id, "name": away}, "score": 3, "isWinner": False},
            "home": {"team": {"id": home_id, "name": home}, "score": 5, "isWinner": True},
        },
    }


def _fake_schedule():
    """
    Three-game fabricated schedule:
      1. Normal game at a correctly-cased, real MLB park (Petco Park).
      2. The UNIQLO bug row: mixed-case sponsor name, correct venue.id=22.
      3. A genuinely neutral-site game (London Series) with a bogus/
         non-MLB venue.id, to prove neutral routing still works via name.
    """
    now = datetime.now(pytz.UTC) - timedelta(hours=2)
    return {
        "dates": [{
            "games": [
                _make_game(1, "Petco Park", 2680, now),
                _make_game(2, "UNIQLO Field at Dodger Stadium", 22, now),
                _make_game(3, "London Stadium", 9999999, now),
            ]
        }]
    }


class TestUniqloVenueMapping:
    """The live ingestion path (Path A) must survive + canonicalize the UNIQLO row."""

    @patch("Simulator.get_game_information.requests.get")
    def test_path_a_fetch_games_canonicalizes_uniqlo_row(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = _fake_schedule()
        mock_get.return_value = mock_response

        from Simulator.get_game_information import fetch_games
        df, games_list = fetch_games(days_ago=3650)

        assert len(df) == 3, "all 3 games (2 valid parks + 1 neutral) should survive Path A's filter"

        uniqlo_row = df[df["gamePk"] == 2].iloc[0]
        assert uniqlo_row["venue.id"] == "22", (
            f"expected canonical id '22', got {uniqlo_row['venue.id']!r} "
            "(mixed-case UNIQLO name must not degrade to 'neutral')"
        )
        assert uniqlo_row["venue.name"] == "Dodger Stadium", (
            f"expected the exact STADIUM_DIMENSIONS key 'Dodger Stadium', "
            f"got {uniqlo_row['venue.name']!r}"
        )

        # No regression: a genuinely neutral-site game still resolves to
        # the neutral id, regardless of whatever bogus venue.id the API sent.
        neutral_row = df[df["gamePk"] == 3].iloc[0]
        assert neutral_row["venue.id"] == DEFAULT_VENUE_ID
        assert neutral_row["venue.name"] == "London Stadium"

        # No regression: an already-correctly-cased park passes through untouched.
        normal_row = df[df["gamePk"] == 1].iloc[0]
        assert normal_row["venue.id"] == "2680"
        assert normal_row["venue.name"] == "Petco Park"


class TestVenueIdToNameInverse:
    """Sanity-check the new VENUE_ID_TO_NAME map added to constants.py."""

    def test_dodger_stadium_id_maps_back_to_canonical_name(self):
        assert VENUE_ID_TO_NAME[22] == "Dodger Stadium"

    def test_every_valid_venue_id_has_a_canonical_name(self):
        assert 22 in VALID_VENUE_IDS
        for vid in VALID_VENUE_IDS:
            assert vid in VENUE_ID_TO_NAME

    def test_canonical_names_prefer_the_non_alias_entry(self):
        # Aliases like 'George M. Steinbrenner Field' and
        # 'UNIQLO FIELD AT DODGER STADIUM' share an id with a canonical park
        # name; the inverse map must resolve to the canonical name so
        # STADIUM_DIMENSIONS lookups succeed.
        assert VENUE_ID_TO_NAME[int(VENUE_NAME_TO_ID['Yankee Stadium'])] == "Yankee Stadium"
        assert VENUE_ID_TO_NAME[int(VENUE_NAME_TO_ID['Guaranteed Rate Field'])] == "Guaranteed Rate Field"
        assert VENUE_ID_TO_NAME[int(VENUE_NAME_TO_ID['Minute Maid Park'])] == "Minute Maid Park"

    def test_neutral_venues_excluded(self):
        for name in NEUTRAL_SITE_VENUES:
            vid = VENUE_NAME_TO_ID.get(name)
            if vid == 'neutral':
                assert 'neutral' not in VENUE_ID_TO_NAME
