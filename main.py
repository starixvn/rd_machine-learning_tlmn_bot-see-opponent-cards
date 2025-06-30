import json
from src.service.bot_see_opponent_cards_2_players import Bot_see_opponent_cards_2_players

bot_see_opponent_card_2_player = Bot_see_opponent_cards_2_players()

def move_2_players(formated_request) -> tuple:

    if isinstance(formated_request, str):
        formated_request = json.loads(formated_request)

    our_card = formated_request['private']['our_card']
    opponent_action = formated_request['private']['opponent_action']
    opponent_actions_card = formated_request['private']['opponent_actions_card']
    our_actions_card = formated_request['private']['our_actions_card']
    opponent_card = formated_request['private']['opponent_card']
    I_play_first = formated_request['private']['I_play_first']

    state = (tuple(our_card), tuple(opponent_action), tuple(opponent_actions_card), 
            tuple(our_actions_card), tuple(opponent_card))
    best_move = bot_see_opponent_card_2_player.choose_best_action_with_max_Q_value(
        state, I_play_first
    )
    return {"best_move": best_move}

if __name__ == "__main__":
    formated_request = {
        "private": {
            "our_card": [2, 4, 5, 11, 12, 20, 22, 24, 25, 27, 32, 47, 49],
            "opponent_action": [10],
            "opponent_actions_card": [10],
            "our_actions_card": [],
            "opponent_card": [8, 14, 19, 21, 23, 26, 30, 31, 34, 39, 45, 50],
            "I_play_first": 0,
        }
    }
    print(f"Best move is: {move_2_players(formated_request)}")

    formated_request = {
        "private": {
            "our_card": [14, 18, 19, 22, 23, 24, 26, 50],
            "opponent_action": [],
            "opponent_actions_card": [8, 16, 33, 40, 51],
            "our_actions_card": [5, 17, 35, 37, 45],
            "opponent_card": [4, 7, 27, 30, 34, 36, 42, 44],
            "I_play_first": 1,
        }
    }
    print(f"Best move is: {move_2_players(formated_request)}")