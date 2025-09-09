import json
from src.service.bot_see_opponent_cards_2_players import Bot_see_opponent_cards_2_players
from src.service.bot_see_opponent_cards_3_players import Bot_see_opponent_cards_3_players
from src.service.bot_2_players_honestly import Bot_2_players_honestly

bot_see_opponent_card_2_player = Bot_see_opponent_cards_2_players()
bot_see_opponent_card_3_player = Bot_see_opponent_cards_3_players()
medium_bot_2_players_honestly = Bot_2_players_honestly(model_dir=r"src/models/Bot_TLMN_solo_medium_level.keras")
hard_bot_2_players_honestly = Bot_2_players_honestly(model_dir=r"src/models/Bot_TLMN_solo_hard_level.keras")

def move_2_players(formated_request):

    # if isinstance(formated_request, str):
    #     formated_request = json.loads(formated_request)

    # our_card = formated_request['private']['our_card']
    # opponent_action = formated_request['private']['opponent_action']
    # opponent_actions_card = formated_request['private']['opponent_actions_card']
    # our_actions_card = formated_request['private']['our_actions_card']
    # opponent_card = formated_request['private']['opponent_card']
    # I_play_first = formated_request['private']['I_play_first']

    # state = (tuple(our_card), tuple(opponent_action), tuple(opponent_actions_card), 
    #         tuple(our_actions_card), tuple(opponent_card))
    # best_move = bot_see_opponent_card_2_player.choose_best_action_with_max_Q_value(
    #     state, I_play_first
    # )
    # return {"best_move": best_move}
    message = "This endpoint is temporary not available. We are updating this model"
    return {"best_move": message}

def cal_deck_remain(our_initial_card, opponent_actions_card):
    return [card for card in range(52) 
            if card not in our_initial_card and card not in opponent_actions_card]

def move_2_players_honestly(formated_request):

    if isinstance(formated_request, str):
        formated_request = json.loads(formated_request)

    our_initial_card = formated_request['private']['our_initial_card']
    our_card = formated_request['private']['our_card']
    opponent_action = formated_request['private']['opponent_action']
    opponent_actions_card = formated_request['private']['opponent_actions_card']
    our_actions_card = formated_request['private']['our_actions_card']
    deck_remain = cal_deck_remain(our_initial_card, opponent_actions_card)
    len_opponent_card = 13 - len(opponent_actions_card)
    print(f"len_opponent_card = {len_opponent_card}")
    I_play_first = formated_request['private']['I_play_first']
    bot_level = formated_request['private'].get('bot_level', 'hard')

    assert len(our_initial_card) == 13, "Our initial card must have 13 cards"
    assert set(our_card).union(set(our_actions_card)) == set(our_initial_card), "Union of our_card and our_actions_card must equal our_initial_card"
    assert not set(our_card).intersection(set(our_actions_card)), "our_card and our_actions_card must be disjoint"
    assert set(opponent_action).issubset(set(opponent_actions_card)), "Opponent action must be subset of opponent actions card"
    assert len(deck_remain) == 39 - len(opponent_actions_card), "Length of deck_remain must be equal to 39 - len(opponent_actions_card)"
    assert not set(deck_remain).intersection(set(our_initial_card)), "deck_remain must not intersect with our_initial_card"
    assert not set(deck_remain).intersection(set(opponent_actions_card)), "deck_remain must not intersect with opponent_actions_card"
    assert isinstance(len_opponent_card, int), "len_opponent_card must be an integer"
    assert 1 <= len_opponent_card <= 13, "len_opponent_card must be between 1 and 13"
    assert I_play_first in [0, 1], "I_play_first must be 0 or 1"
    assert bot_level in ["hard", "medium"], "bot_level field must be either 'hard' or 'medium'"
    
    if bot_level == "hard":
        bot_2_players_honestly_cls = hard_bot_2_players_honestly
    else:
        bot_2_players_honestly_cls = medium_bot_2_players_honestly
    print(f"The model directory is: {bot_2_players_honestly_cls.model_dir}")

    state = (tuple(our_card), tuple(opponent_action), tuple(opponent_actions_card), 
            tuple(our_actions_card), tuple(deck_remain), len_opponent_card)
    best_move = bot_2_players_honestly_cls.choose_best_action_with_max_Q_value(
        state, I_play_first
    )
    return {"best_move": best_move}

def compress_to_state(player_card, player_action, turn):
    n_player = len(player_card)
    skip_int_encode = bot_see_opponent_card_3_player.skip_int_encode
    player_card_encode = tuple(tuple(player_card[f"player_{i}"]) for i in range(1, n_player + 1))
    player_action_encode = tuple(
        tuple(player_action[f"player_{i}"]) if player_action[f"player_{i}"] != skip_int_encode else skip_int_encode
        for i in range(2, n_player + 1)
    )
    state = (player_card_encode, player_action_encode, turn)
    return state

def convert_initial_player_card(initial_player_card):
    rename_map = {old_key: int(old_key[-1]) for old_key in initial_player_card.keys()}
    rename_initial_player_card = {rename_map[k]: v for k, v in initial_player_card.items()}
    assert set(rename_initial_player_card.keys()) == set(range(1, len(initial_player_card)+1))
    assert all(len(cards) == 13 for cards in rename_initial_player_card.values())
    return rename_initial_player_card

def move_3_players(formated_request):

    # if isinstance(formated_request, str):
    #     formated_request = json.loads(formated_request)

    # initial_player_card = formated_request['private']['initial_player_card']
    # player_card = formated_request['private']['player_card']
    # player_action = formated_request['private']['player_action']
    # turn = formated_request['private']['turn']
    # who_play_first = formated_request['private']['who_play_first']
    # player1_skip = formated_request['private']['player1_skip']
    # assert set(initial_player_card.keys()) == set(player_card.keys()) == {"player_1", "player_2", "player_3"}
    # assert all(len(cards) == 13 for cards in initial_player_card.values())
    # assert set(player_action.keys()) == {"player_2", "player_3"}
    # assert player1_skip in [0, 1]

    # state = compress_to_state(player_card, player_action, turn)
    # initial_player_card = convert_initial_player_card(initial_player_card)
    # best_move = bot_see_opponent_card_3_player.choose_best_action_with_max_Q_value(
    #     state, initial_player_card, who_play_first, player1_skip
    # )
    # return {"best_move": best_move}
    message = "This endpoint is temporary not available. We are updating this model"
    return {"best_move": message}

if __name__ == "__main__":
    # formated_request = {
    #     "private": {
    #         "our_card": [2, 4, 5, 11, 12, 20, 22, 24, 25, 27, 32, 47, 49],
    #         "opponent_action": [10],
    #         "opponent_actions_card": [10],
    #         "our_actions_card": [],
    #         "opponent_card": [8, 14, 19, 21, 23, 26, 30, 31, 34, 39, 45, 50],
    #         "I_play_first": 0
    #     }
    # }
    # print(f"Best move is: {move_2_players(formated_request)}")

    # formated_request = {
    #     "private": {
    #         "our_card": [14, 18, 19, 22, 23, 24, 26, 50],
    #         "opponent_action": [],
    #         "opponent_actions_card": [8, 16, 33, 40, 51],
    #         "our_actions_card": [5, 17, 35, 37, 45],
    #         "opponent_card": [4, 7, 27, 30, 34, 36, 42, 44],
    #         "I_play_first": 1
    #     }
    # }
    # print(f"Best move is: {move_2_players(formated_request)}")

    # formated_request = {
    #     "private": {
    #         "initial_player_card": {"player_1": [20, 21, 22, 23, 25, 26, 32, 33, 35, 38, 42, 43, 45], 
    #                                 "player_2": [1, 6, 9, 11, 12, 13, 15, 24, 29, 34, 37, 40, 50], 
    #                                 "player_3": [0, 4, 5, 7, 8, 16, 19, 28, 36, 46, 47, 48, 51]},
    #         "player_card": {"player_1": [20, 21, 22, 23, 25, 26, 33, 35, 38, 42, 43, 45], 
    #                         "player_2": [1, 6, 9, 11, 12, 13, 15, 24, 29, 34, 37, 40], 
    #                         "player_3": [0, 4, 5, 7, 8, 16, 19, 28, 36, 46, 47, 48, 51]},
    #         "player_action": {"player_2": [50], "player_3": []},
    #         "turn": 2, # Player id turn: 1, 2, 3
    #         "who_play_first": 1, # Player id: 1, 2, 3
    #         "player1_skip": 0
    #     }
    # }
    # print(f"Best move is: {move_3_players(formated_request)}")

    # formated_request = {
    #     "private": {
    #         "initial_player_card": {"player_1": [0, 10, 11, 14, 17, 20, 24, 25, 28, 38, 49, 50, 51], 
    #                                 "player_2": [3, 6, 7, 8, 12, 16, 22, 26, 29, 34, 37, 45, 46], 
    #                                 "player_3": [1, 4, 15, 23, 32, 35, 36, 40, 41, 42, 43, 47, 48]},
    #         "player_card": {"player_1": [11, 14, 17, 51], 
    #                         "player_2": [3, 7, 8, 12, 16, 22, 26, 29, 34, 37, 45, 46], 
    #                         "player_3": [32, 35, 41, 42]},
    #         "player_action": {"player_2": [], "player_3": []},
    #         "turn": 1,
    #         "who_play_first": 1,
    #         "player1_skip": 0
    #     }
    # }
    # print(f"Best move is: {move_3_players(formated_request)}")

    formated_request = {
        "private": {
            "our_initial_card": [3, 13, 14, 18, 19, 23, 25, 28, 33, 34, 48, 49, 51],
            "our_card": [3, 48, 49],
            "opponent_action": [],
            "opponent_actions_card": [1, 4, 6, 7, 8, 10, 16, 31, 39, 40, 42, 47],
            "our_actions_card": [13, 14, 18, 19, 23, 25, 28, 33, 34, 51],
            "I_play_first": 0
        }
    }
    print(f"Best move 2 players honestly is: {move_2_players_honestly(formated_request)}")

    formated_request = {
        "private": {
            "our_initial_card": [3, 13, 14, 18, 19, 23, 25, 28, 33, 34, 48, 49, 51],
            "our_card": [3, 23, 25, 28, 34, 49],
            "opponent_action": [],
            "opponent_actions_card": [1, 4, 6, 7, 8, 10, 16, 31, 39, 40, 42, 47],
            "our_actions_card": [13, 14, 18, 19, 33, 48, 51],
            "I_play_first": 0
        }
    }
    print(f"Best move 2 players honestly is: {move_2_players_honestly(formated_request)}")

    formated_request = {
        "private": {
            "our_initial_card": [3, 13, 14, 18, 19, 23, 25, 28, 33, 34, 48, 49, 51],
            "our_card": [3, 13, 14, 18, 19, 23, 25, 28, 33, 34],
            "opponent_action": [],
            "opponent_actions_card": [1, 4, 6, 7, 8, 10, 16, 31, 39, 40, 42, 47],
            "our_actions_card": [48, 49, 51],
            "I_play_first": 0
        }
    }
    print(f"Best move 2 players honestly is: {move_2_players_honestly(formated_request)}")

    formated_request = {
        "private": {
            "our_initial_card": [1, 6, 7, 8, 9, 10, 11, 12, 25, 29, 36, 49, 51],
            "our_card": [49, 51],
            "opponent_action": [],
            "opponent_actions_card": [4, 5, 18, 19, 20, 24, 28, 30, 32, 33, 35, 44],
            "our_actions_card": [1, 6, 7, 8, 9, 10, 11, 12, 25, 29, 36],
            "I_play_first": 1
        }
    }
    print(f"Best move 2 players honestly is: {move_2_players_honestly(formated_request)}")
    print("\n\n")

    formated_request = {
        "private": {
            "our_initial_card": [3, 20, 21, 22, 23, 32, 35, 36, 39, 40, 41, 42, 51],
            "our_card": [23, 32, 35, 39, 41, 42, 51],
            "opponent_action": [],
            "opponent_actions_card": [0, 1, 7, 8, 9, 11, 13, 26, 29, 34, 38, 49],
            "our_actions_card": [3, 20, 21, 22, 36, 40],
            "I_play_first": 0
        }
    }
    output = move_2_players_honestly(formated_request)
    print(f"Best move 2 players honestly is: {output}")
    from src.tienlen_libs import Card
    Card.print_pretty_cards(output["best_move"])