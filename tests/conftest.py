import pytest
from src.service.bot_2_players_honestly import Bot_2_players_honestly

@pytest.fixture
def bot_2_players_honestly_cls():
    return Bot_2_players_honestly()