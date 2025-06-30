import time
from src.tienlen_libs import Card, ActionGenerator, Utils
import os
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.models import load_model


class Bot_see_opponent_cards_2_players(object):
    def __init__(self):
        self.action_generator = ActionGenerator()
        self.MLP_model = load_model(r"src/models/TLMN_see_opponent_card_2_players.keras")
     
    def generate_one_hot_encoding(self, cards: tuple[int], dimension: int = 52):
        return tuple(1 if i in cards else 0 for i in range(dimension))
    
    def generate_action_encoding(self, action: tuple[int]):
        action_int_combo_type = Utils.get_int_combo_type(action)
        action_int_combo_type_encoding = self.generate_one_hot_encoding((action_int_combo_type,),
                                                                        dimension=19)
        assert action_int_combo_type_encoding.count(1) == 1        
        max_card = max(action) if len(action) > 0 else -1
        max_card = (max_card + 1) / 52
        return action_int_combo_type_encoding + (max_card,)
    
    def encode_state(self, state):
        player2_card, player1_action, actions_card_player1, actions_card_player2, player1_card = state
        player2_card_encoding = self.generate_one_hot_encoding(player2_card)
        player1_action_encoding = self.generate_action_encoding(player1_action)
        actions_card_player1_encoding = self.generate_one_hot_encoding(actions_card_player1)
        actions_card_player2_encoding = self.generate_one_hot_encoding(actions_card_player2)
        player1_card_encoding = self.generate_one_hot_encoding(player1_card)

        state_encode = player2_card_encoding + player1_action_encoding + \
                        actions_card_player1_encoding + actions_card_player2_encoding + \
                        player1_card_encoding
        return state_encode
    
    def predict_Q_value(self, list_state_action_encode):
        state_action_array = np.array(list_state_action_encode, dtype=np.float64)
        assert all(x >= 0 and x <= 1 for x in state_action_array.flatten()), "Input must contain only 0s and 1s"
        Q_values = self.MLP_model.predict(state_action_array, verbose=1).flatten()
        return Q_values

    def choose_best_action_with_max_Q_value(self, state, who_play_first):
        state_encoding = self.encode_state(state)
        assert who_play_first in [1, 2]
        player2_play_first = 1 if who_play_first == 2 else 0

        player2_card, player1_action, _, _, _ = state
        list_valid_action_player2 = self.action_generator.get_list_valid_action(player2_card, player1_action)
        list_state_action_encode = []
        for action in list_valid_action_player2:
            action_encoding = self.generate_action_encoding(action)
            state_action_encode = state_encoding + action_encoding + (player2_play_first,)
            list_state_action_encode.append(state_action_encode)

        Q_values = self.predict_Q_value(list_state_action_encode)
        action_to_Q_value = dict(zip(list_valid_action_player2, Q_values))
        for action, Q_value in action_to_Q_value.items():
            print("Action: ", end=""); Card.print_pretty_cards(action, end=" => ")
            print(f"Q_value = {float(Q_value)}")
        player2_action = max(action_to_Q_value.items(), key=lambda x: x[1])[0]
        return player2_action


if __name__ == "__main__":
    bot = Bot_see_opponent_cards_2_players()
    start_time = time.time()
    print("Case 1:")
    player2_card = [14, 18, 19, 22, 23, 24, 26, 50]
    player1_action = []
    actions_card_player1 = [8, 16, 33, 40, 51]
    actions_card_player2 = [5, 17, 35, 37, 45]
    player1_card = [4, 7, 27, 30, 34, 36, 42, 44]
    state = (tuple(player2_card), tuple(player1_action), tuple(actions_card_player1), 
            tuple(actions_card_player2), tuple(player1_card))
    bot.choose_best_action_with_max_Q_value(state, who_play_first=2)

    print(f"\nCase 2:")
    player2_card = [2, 4, 5, 11, 12, 20, 22, 24, 25, 27, 32, 47, 49]
    player1_action = [10]
    actions_card_player1 = [10]
    actions_card_player2 = []
    player1_card = [8, 14, 19, 21, 23, 26, 30, 31, 34, 39, 45, 50]
    state = (tuple(player2_card), tuple(player1_action), tuple(actions_card_player1), 
            tuple(actions_card_player2), tuple(player1_card))
    bot.choose_best_action_with_max_Q_value(state, who_play_first=1)