import time
from src.tienlen_libs import Card, ActionGenerator, Utils
from src.tienlen_libs.constant import ComboType
import os
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.models import load_model


class Bot_see_opponent_cards_3_players(object):
    def __init__(self):
        self.utils = Utils()
        self.action_generator = ActionGenerator()
        self.MLP_model = load_model(r"src/models/TLMN_see_opponent_cards_3_players.keras")
        self.n_player = 3
        self.skip_int_encode = -1
     
    def generate_one_hot_encoding(self, cards: tuple[int], dimension: int = 52):
        return tuple(1 if i in cards else 0 for i in range(dimension))

    def generate_action_encoding(self, action: tuple[int]):
        skip_int_encode = -1
        if action == skip_int_encode:
            action_int_combo_type = 19
        else:
            action_int_combo_type = Utils.get_int_combo_type(action)
            assert action_int_combo_type < 19
        action_int_combo_type_encoding = self.generate_one_hot_encoding((action_int_combo_type,),
                                                                        dimension=20)
        assert action_int_combo_type_encoding.count(True) == 1        
        max_card = max(action) if (action != skip_int_encode and len(action) > 0) else -1
        max_card = (max_card + 1) / 52
        return action_int_combo_type_encoding + (max_card,)

    def generate_player_card_encoding(self, player_card: tuple[tuple[int]]):
        return tuple(
            encoding
            for player_id_card in player_card
            for encoding in self.generate_one_hot_encoding(player_id_card)
        )

    def generate_player_action_encoding(self, player_action: tuple[tuple[int]]):
        return tuple(
            encoding
            for player_id_action in player_action
            for encoding in self.generate_action_encoding(player_id_action)
        )

    def generate_actions_card_player_encoding(self, player_card, initial_player_card):
        n_player = len(initial_player_card)
        actions_card_player_encoding = tuple()
        for i in range(n_player):
            player_id_card = player_card[i]
            initial_player_id_card = initial_player_card[i+1]
            actions_card_player_id = tuple([
                card for card in initial_player_id_card if card not in player_id_card
            ])
            assert len(player_id_card) + len(actions_card_player_id) == 13
            assert set(player_id_card).union(set(actions_card_player_id)) == set(initial_player_id_card)
            actions_card_player_encoding += self.generate_one_hot_encoding(actions_card_player_id)
        return actions_card_player_encoding
    
    def encode_state(self, state, initial_player_card):
        player_card, player_action, turn = state
        assert len(player_card) == self.n_player
        assert len(player_action) == self.n_player - 1
        assert turn in range(1, self.n_player + 1)
        player_card_encoding = self.generate_player_card_encoding(player_card)
        player_action_encoding = self.generate_player_action_encoding(player_action)
        actions_card_player_encoding = self.generate_actions_card_player_encoding(
                                                player_card, initial_player_card)
        turn_encoding = self.generate_one_hot_encoding((turn-1,), dimension=self.n_player)
        assert turn_encoding.count(True) == 1

        state_encode = player_card_encoding + player_action_encoding + \
                        actions_card_player_encoding + turn_encoding
        return state_encode
    
    def is_deuce_or_pair_deuce(self, player_id_action):
        return self.utils.get_combo_type(player_id_action) in [ComboType.SINGLE_DEUCE, ComboType.PAIR_DEUCE]
    
    def get_list_valid_action_in_skip_condition(self, player1_card, last_action):
        list_possible_action = []
        if self.is_deuce_or_pair_deuce(last_action):
            list_valid_action = self.action_generator.get_list_valid_action(player1_card, last_action)
            for valid_action in list_valid_action:
                if self.utils.get_combo_type(valid_action) == ComboType.DOUBLE_SEQUENCE_FOUR:
                    list_possible_action.append(valid_action)
        list_possible_action.append(self.skip_int_encode)
        return list_possible_action

    def get_list_valid_action_of_player1(self, state, initial_player_card, skip):
        player_card, player_action, turn = state
        player1_card = player_card[0]
        assert set(player1_card).issubset(set(initial_player_card[1]))
        assert turn in range(1, self.n_player + 1) and len(player_action) == 2
        last_action = player_action[turn-2] if turn != 1 else tuple()
        if not skip:
            list_valid_action_player1 = self.action_generator.get_list_valid_action(player1_card, last_action)
        else:
            assert turn != 1
            list_valid_action_player1 = self.get_list_valid_action_in_skip_condition(player1_card, last_action)
        return list_valid_action_player1

    def predict_Q_value(self, list_state_action_encode):
        state_action_array = np.array(list_state_action_encode, dtype=np.float64)
        assert all(x >= 0 and x <= 1 for x in state_action_array.flatten()), "Input must contain only 0s and 1s"
        Q_values = self.MLP_model.predict(state_action_array, verbose=1).flatten()
        return Q_values

    def choose_best_action_with_max_Q_value(self, state, initial_player_card, who_play_first, skip):
        assert isinstance(initial_player_card, dict) and isinstance(state, tuple)
        assert who_play_first in range(1, self.n_player + 1) and skip in [0, 1]
        state_encoding = self.encode_state(state, initial_player_card)
        who_play_first_encoding = self.generate_one_hot_encoding((who_play_first-1,), 
                                                                dimension=self.n_player)
        assert who_play_first_encoding.count(True) == 1

        list_valid_action_player1 = self.get_list_valid_action_of_player1(state, initial_player_card, skip)
        if len(list_valid_action_player1) == 1: return list_valid_action_player1[0]
        list_state_action_encode = []
        for action in list_valid_action_player1:
            action_encoding = self.generate_action_encoding(action)
            state_action_encode = state_encoding + action_encoding + who_play_first_encoding
            list_state_action_encode.append(state_action_encode)

        Q_values = self.predict_Q_value(list_state_action_encode)
        action_to_Q_value = dict(zip(list_valid_action_player1, Q_values))
        for action, Q_value in action_to_Q_value.items():
            print("Action: ", end="")
            if action != self.skip_int_encode:
                Card.print_pretty_cards(action, end=" => ")
            else:
                print(action, end=" => ")
            print(f"Q_value = {float(Q_value)}")
        player2_action = max(action_to_Q_value.items(), key=lambda x: x[1])[0]
        return player2_action


if __name__ == "__main__":
    pass