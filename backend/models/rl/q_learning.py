import numpy as np
import pandas as pd
from pathlib import Path
import pickle

# GLOBAL HYPERPARAMETERS FOR REINFORCEMENT LEARNING SETUP
NUM_EPISODES = 10000
NUM_ACTIONS = 2 # do not interrupt or interrupt
GAMMA = 0.95 # how much we weigh future rewards vs. immediate rewards
EPSILON_START = 0.3 # initial exploration rate for epsilon-greedy
EPSILON_DECAY = 0.999 # decay rate for epsilon
EPSILON_MIN = 0.05 # ensures we keep exploring
INTERRUPT_COST = 0.1 # penalty for choosing interrupt

def initialize_Q() -> (dict[int, np.ndarray], dict[int, np.ndarray]):
    """
    Initialize an empty Q and N table
    
    Parameters:
        None
    
    Returns:
        Q (dict[int, np.ndarray]): The Q-table of a state id to their respective Q value for each action.
                                   Each state id posesses an array of length NUM_ACTIONS
        N (dict[int, np.ndarray]): Tracks the count of how many times each action has been visites.
                                   N[state_id][a] is how many times (state_id, a) has been visited
    """
    Q = {}
    N = {}
    
    return Q, N

def save_Q(Q: dict, N: dict, num_episodes: int=NUM_EPISODES, epsilon_decay: float=EPSILON_DECAY) -> None:
    """
    Saves the current Q and N tables in a singular pickle file
    
    Parameters:
        Q (dict[int, np.ndarray]): The Q-table of a state id to their respective Q value for each action.
        N (dict[int, np.ndarray]): Tracks the count of how many times each action has been visites.
        num_episodes (int): the number of episodes used to run. Used for naming the pickle file
        epsilon_decay (float): the decay rate for epsilon. Used for naming the pickle file
        
    Returns:
        None 
    """
    data = {"Q": Q, "N": N}
    filename = f"Q_table_{num_episodes}_{epsilon_decay}.pickle"
    
    with open(filename, "wb") as handle:
        pickle.dump(data, handle)
        
def load_Q(num_episodes: int=NUM_EPISODES, epsilon_decay: float=EPSILON_DECAY):
    """
    Load the Q and N tables from the matching file with parameters of num_episodes and epsilon_decay
    
    Parameters:
        num_episodes (int): the number of episodes used to run. Used for the pickle file name
        epsilon_decay (float): the decay rate for epsilon. Used for the pickle file name
    
    Return:
        Q (dict[int, np.ndarray]): The Q-table of a state id to their respective Q value for each action.
                                   Each state id posesses an array of length NUM_ACTIONS
        N (dict[int, np.ndarray]): Tracks the count of how many times each action has been visites.
                                   N[state_id][a] is how many times (state_id, a) has been visited
    """
    filename = f"Q_table_{num_episodes}_{epsilon_decay}.pickle"
    
    # Try to find and load the pickle file given the parameters
    try:
        with open(filename, "rb") as handle:
            data = pickle.load(handle)
    
        Q = data.get("Q", {})
        N = data.get("N", {})
        return Q, N
    
    # If no pickle file is found, return a newly initialized Q and N table
    except FileNotFoundError:
        print(f"{filename} not found, returning an initialization of Q and N")
        return initialize_Q()

def hash_state(state: np.ndarray) -> int:
    """
    Turn continuous features into a discrete state id
    
    Parameters:
        state (np.ndarray): [nervous_score, eye_contact, speech_rate_wpm, filler_rate, silence_duration]
        
    Returns:
        int: packs each feature into a singular integer
    """
    
    nervous, eye, speech, filler, silence = state
    
    # nervous bin information:
    #   0 -> low nervousness
    #   1 -> medium nervousness
    #   2 -> high nervousness 
    if nervous < 0.3:
        nervous_bin = 0
    elif nervous < 0.6:
        nervous_bin = 1
    else:
        nervous_bin = 2

    # eye contact bin information
    #   0 -> poor eye contact
    #   1 -> medium eye contact
    #   2 -> good eye contact
    if eye < 0.3:
        eye_bin = 0
    elif eye < 0.6:
        eye_bin = 1
    else:
        eye_bin = 2 

    # speech rate (wpm) bin information
    #   0 -> slow
    #   1 -> normal
    #   2 -> fast
    #   3 -> very fast
    if speech < 90:
        speech_bin = 0
    elif speech < 160:
        speech_bin = 1
    elif speech < 220:
        speech_bin = 2
    else:
        speech_bin = 3

    # filler word rate bin information (may have to be altered later)
    #   0 -> low use of filler words
    #   1 -> medium use of filler words
    #   2 -> frequent use of filler words
    #   3 -> very frequent use of filler words
    if filler < 0.05:
        filler_bin = 0
    elif filler < 0.15:
        filler_bin = 1
    elif filler < 0.3:
        filler_bin = 2
    else:
        filler_bin = 3

    # silence duration bin information; how long since last word was spoken (may have to be altered later)
    #   0 -> short silence (confidentely speaking)
    #   1 -> normal silence (most likely indicating a new paragraph starting, or answering a new question)
    #   2 -> long silence (most likely getting stuck; thinking about an answer) 
    if silence < 0.2:
        silence_bin = 0
    elif silence < 1.0:
        silence_bin = 1
    else:
        silence_bin = 2

    # multiple by the number of options bins each number has; bin lengths: nervous (3), eye (3), speech (4), filler (4), silence (3)
    # packs the bins into a singular state integer
    state_id = nervous_bin
    state_id = state_id * 3 + eye_bin
    state_id = state_id * 4 + speech_bin
    state_id = state_id * 4 + filler_bin
    state_id = state_id * 3 + silence_bin
    
    return int(state_id)

def epsilon_greedy_action(Q: dict, state_id: int, num_actions: int=NUM_ACTIONS, epsilon: float=EPSILON_START) -> int:
	"""
	Chooses an action based on the epsilon greedy exploration policy
	
	Parameters:
		Q (dict[int, np.ndarray]): The Q-table of a state id to their respective Q value for each action.
                                   Each state id posesses an array of length NUM_ACTIONS
		state_id (int): The hashed state representation of the environment
		num_actions (int): The number of currently available actions 
		epsilon (float): The probability of selecting a random action
  
	Returns:
		int: The index of the chosen action
	"""
		
	# Choose a random action if the probability is hit
	if np.random.random() < epsilon:
		return np.random.randint(num_actions)
		
	q_values = Q.get(state_id, np.zeros(num_actions, dtype=float))
		
	return int(np.argmax(q_values))

def compute_reward(prev_state: np.ndarray, curr_state: np.ndarray, prev_action: int) -> float:
    """
    Compute a reward for transitioning from the previous state to the current state given that we took the previous action.
    We can define a user confidence score dictated based off of their nervousness, eye contact, filler words, and pauses. 
    
    Parameters:
        prev_state (np.ndarray): The previous hashed state representation of the environment
        curr_state (np.ndarray): The current hashed state representation of the environment
        prev_action (int): the previous action index taken
        
    Returns: 
        the reward observed after transitioning
    """
    # unpack our specific values for the previousd and current states
    prev_nervous, prev_eye, prev_speech, prev_filler, prev_silence = prev_state
    curr_nervous, curr_eye, curr_speech, curr_filler, curr_silence = curr_state
    
    # -----PREVIOUS STATE INFORMATION-----
    # nervous => 0 (calm) becomes 1.0 reward; 1 (very nervous) becomes 0.0 reward; clamped to [0,1] to ensure same units
    prev_nervous_fixed = 1.0 - float(np.clip(prev_nervous, 0.0, 1.0)) # 1 = calm, 0 = nervous
    
    # eye contact => already has 1.0 for good eye contact and 0.0 for bad eye contact; already clamped, but done again just in case
    prev_eye_fixed = float(np.clip(prev_eye, 0.0, 1.0)) #
    
    # speech rate => ideal rate we are making ~ 150 wpm. we penalize deviation from that
    prev_speech_clamped = np.clip(prev_speech, 60, 240) # clamps to [60, 240]
    prev_speech_dev = np.abs(prev_speech_clamped - 150.0) # restricts to a normal range
    prev_speech_term = 1.0 - (prev_speech_dev / 90.0)
    prev_speech_fixed = np.clip(prev_speech_term, 0.0, 1.0)
    
    # filler rate => 0 is best
    prev_filler_clamped = np.clip(prev_filler, 0.0, 0.4)
    prev_filler_term = 1.0 - (prev_filler_clamped / 0.4)
    prev_filler_fixed = np.clip(prev_filler_term, 0.0, 1.0)
    
    # silence duration => short pauses are okay, very long is bad
    prev_silence_clamped = np.clip(prev_silence, 0.0, 3.0)
    prev_silence_term = 1.0 - (prev_silence_clamped / 3.0)
    prev_silence_fixed = np.clip(prev_silence_term, 0.0, 1.0)
    
    # weighted previous state confidence score
    # 0.30 weight => nervous and eye are more serious factors
    # 0.15 weight => speech an filler; less serious but not the least serious
    # 0.10 weight => silence timing; can be rather unhelpful/unpredictable for various interviews
    prev_confidence = (
        0.30 * prev_nervous_fixed +
        0.30 * prev_eye_fixed + 
        0.15 * prev_speech_term + 
        0.15 * prev_filler_term +
        0.10 * prev_silence_term
    )
    # -----------------------------------
     
    # -----CURRENT STATE INFORMATION-----
    # nervous => 0 (calm) becomes 1.0 reward; 1 (very nervous) becomes 0.0 reward; clamped to [0,1] to ensure same units
    curr_nervous_fixed = 1.0 - float(np.clip(curr_nervous, 0.0, 1.0)) # 1 = calm, 0 = nervous
    
    # eye contact => already has 1.0 for good eye contact and 0.0 for bad eye contact; already clamped, but done again just in case
    curr_eye_fixed = float(np.clip(curr_eye, 0.0, 1.0)) #
    
    # speech rate => ideal rate we are making ~ 150 wpm. we penalize deviation from that
    curr_speech_clamped = np.clip(curr_speech, 60, 240) # clamps to [60, 240]
    curr_speech_dev = np.abs(curr_speech_clamped - 150.0) # restricts to a normal range
    curr_speech_term = 1.0 - (curr_speech_dev / 90.0)
    curr_speech_fixed = np.clip(curr_speech_term, 0.0, 1.0)
    
    # filler rate => 0 is best
    curr_filler_clamped = np.clip(curr_filler, 0.0, 0.4)
    curr_filler_term = 1.0 - (curr_filler_clamped / 0.4)
    curr_filler_fixed = np.clip(curr_filler_term, 0.0, 1.0)
    
    # silence duration => short pauses are okay, very long is bad
    curr_silence_clamped = np.clip(curr_silence, 0.0, 3.0)
    curr_silence_term = 1.0 - (curr_silence_clamped / 3.0)
    curr_silence_fixed = np.clip(curr_silence_term, 0.0, 1.0)
    
    # weighted currious state confidence score
    # 0.30 weight => nervous and eye are more serious factors
    # 0.15 weight => speech an filler; less serious but not the least serious
    # 0.10 weight => silence timing; can be rather unhelpful/unpredictable for various interviews
    curr_confidence = (
        0.30 * curr_nervous_fixed +
        0.30 * curr_eye_fixed + 
        0.15 * curr_speech_term + 
        0.15 * curr_filler_term +
        0.10 * curr_silence_term
    )
    # -----------------------------------
    
    # how did the user confidence improve; calculate the reward (delta of current and previous confidence); positive is the confidence went up, negative is it went down
    reward = curr_confidence - prev_confidence 
    
    # checks for how low the old confidence was clipped to [0, 1]
    prev_confidence_clipped = 1.0 - np.clip(prev_confidence, 0.0, 1.0) 
    
    if prev_action == 1:
        # if we interrupted before, then give an extra bonus for interrupting when the user was doing poorly; showing it worked (the worse it was, the bigger the bonus is)
        reward -= INTERRUPT_COST
        
        if prev_confidence_clipped > 0.6:
            reward += 0.3 * prev_confidence_clipped
    else:
        # if we did not interrup before, then given a penalty for not interrupting when the user was doing bad; leans us towards an interruption (the better it is, the less the bonus is)
        reward += 0.1 * (1.0 - prev_confidence_clipped)
        
    return float(reward)

def update_Q(Q: dict, N: dict, state_id: int, next_state_id: int, action: int, reward: float, gamma: float) -> None:
    """
    Perform a singular Q-learning update using the update rule formula
    
    Parameters:
        Q (dict[int, np.ndarray]): The Q-table of a state id to their respective Q value for each action.
                                   Each state id posesses an array of length NUM_ACTIONS
        N (dict[int, np.ndarray]): Tracks the count of how many times each action has been visites.
                                   N[state_id][a] is how many times (state_id, a) has been visited
        state_id (int): The hashed state representation of the environment
        next_state_id (int): The next hashed state representation of the environment
        action (int): action index taken
        reward (float): the reward observed after transitioning to next_state_id
        gamma (float): The discount factor
        
    Returns:
        None
    """
    
    if state_id not in Q:
        Q[state_id] = np.zeros(NUM_ACTIONS, dtype=float)
        N[state_id] = np.zeros(NUM_ACTIONS, dtype=int)
    if next_state_id not in Q:
        Q[next_state_id] = np.zeros(NUM_ACTIONS, dtype=float)
        N[next_state_id] = np.zeros(NUM_ACTIONS, dtype=int)
        
    eta = 1.0 / (1.0 + N[state_id][action]) 
 
    # Calculate the temporal difference for our Q-learning update rule formula
    temporal_difference = reward + gamma * np.max(Q[next_state_id])
 
    # use Q-learning update rule
    Q[state_id][action] = Q[state_id][action] + eta * (temporal_difference - Q[state_id][action])
 
    N[state_id][action] += 1