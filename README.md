This project for containing the bot for TLMN with the scenario of seeing opponent cards

For 2 players, the request API example are:

{
    "private": {
        "our_card": [14, 18, 19, 22, 23, 24, 26, 50],
        "opponent_action": [],
        "opponent_actions_card": [8, 16, 33, 40, 51],
        "our_actions_card": [5, 17, 35, 37, 45],
        "opponent_card": [4, 7, 27, 30, 34, 36, 42, 44],
        "I_play_first": 1
    }
}

{
    "private": {
        "our_card": [2, 4, 5, 11, 12, 20, 22, 24, 25, 27, 32, 47, 49],
        "opponent_action": [10],
        "opponent_actions_card": [10],
        "our_actions_card": [],
        "opponent_card": [8, 14, 19, 21, 23, 26, 30, 31, 34, 39, 45, 50],
        "I_play_first": 0
    }
}

** I_play_first is only 0 or 1 value, determine whether I play the game first or not