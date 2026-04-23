"""
Finite State Machine simulation infrastructure for anomaly detection.

Provides a base FSM class and three concrete implementations:
  1. TrafficLightFSM     — 4 states, input-independent fixed cycle
  2. BitPatternDetectorFSM — 5 states, binary input, detects "1011"
  3. VendingMachineFSM   — 6 states, coin inputs (nickel/dime)
"""

from abc import ABC, abstractmethod


class FSM(ABC):
    """Abstract base class for deterministic finite state machines (Moore machines)."""

    def __init__(self, state_names, initial_state):
        self.state_names = state_names
        self.num_states = len(state_names)
        self.initial_state = initial_state
        self.current_state = initial_state

    def reset(self):
        """Set current state back to the initial state."""
        self.current_state = self.initial_state

    @abstractmethod
    def step(self, input_val=None):
        """Apply transition function, update current_state, return new state ID."""

    def run(self, input_sequence):
        """Run FSM on an input sequence and return the full state trace.

        Returns a list of state IDs of length len(input_sequence) + 1,
        starting with the initial state.
        """
        self.reset()
        trace = [self.current_state]
        for inp in input_sequence:
            self.step(inp)
            trace.append(self.current_state)
        return trace

    def state_name(self, state_id):
        """Return the human-readable name for a state ID."""
        return self.state_names[state_id]

    def trace_names(self, trace):
        """Convert a list of state IDs to human-readable state names."""
        return [self.state_names[s] for s in trace]

    def get_legal_transitions(self):
        """Return the set of all legal (state, next_state) pairs.

        Subclasses should override if they can provide this more efficiently,
        but this default inspects the transition table attribute if present.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# FSM 1 — Traffic Light Controller
# ---------------------------------------------------------------------------

class TrafficLightFSM(FSM):
    """4-state traffic light with fixed cycle: RED → RED_YELLOW → GREEN → YELLOW → RED.

    Input-independent: step() ignores its argument and always advances the cycle.
    """

    RED, RED_YELLOW, GREEN, YELLOW = 0, 1, 2, 3

    def __init__(self):
        super().__init__(
            state_names=["RED", "RED_YELLOW", "GREEN", "YELLOW"],
            initial_state=self.RED,
        )
        self._transitions = {
            self.RED: self.RED_YELLOW,
            self.RED_YELLOW: self.GREEN,
            self.GREEN: self.YELLOW,
            self.YELLOW: self.RED,
        }

    def step(self, input_val=None):
        self.current_state = self._transitions[self.current_state]
        return self.current_state

    def get_legal_transitions(self):
        return {(s, ns) for s, ns in self._transitions.items()}


# ---------------------------------------------------------------------------
# FSM 2 — Serial Bit-Pattern Detector ("1011")
# ---------------------------------------------------------------------------

class BitPatternDetectorFSM(FSM):
    """5-state Moore machine that detects the bit sequence "1011".

    States:
      S0 — no match progress (initial)
      S1 — matched "1"
      S2 — matched "10"
      S3 — matched "101"
      S4 — matched "1011" (pattern detected)
    """

    S0, S1, S2, S3, S4 = 0, 1, 2, 3, 4

    def __init__(self):
        super().__init__(
            state_names=["S0", "S1", "S2", "S3", "S4"],
            initial_state=self.S0,
        )
        # _transitions[current_state][input_bit] = next_state
        self._transitions = {
            self.S0: {0: self.S0, 1: self.S1},
            self.S1: {0: self.S2, 1: self.S1},
            self.S2: {0: self.S0, 1: self.S3},
            self.S3: {0: self.S2, 1: self.S4},
            self.S4: {0: self.S2, 1: self.S1},
        }

    def step(self, input_val=None):
        self.current_state = self._transitions[self.current_state][input_val]
        return self.current_state

    def get_legal_transitions(self):
        pairs = set()
        for state, trans in self._transitions.items():
            for next_state in trans.values():
                pairs.add((state, next_state))
        return pairs


# ---------------------------------------------------------------------------
# FSM 3 — Vending Machine Controller
# ---------------------------------------------------------------------------

class VendingMachineFSM(FSM):
    """6-state vending machine that accumulates coins to 20¢ then dispenses.

    States: IDLE → FIVE → TEN → FIFTEEN → DISPENSE → CHANGE → IDLE
    Inputs: NICKEL (5), DIME (10), or NONE (0).
    DISPENSE and CHANGE are automatic transitions (input is ignored).
    """

    IDLE, FIVE, TEN, FIFTEEN, DISPENSE, CHANGE = 0, 1, 2, 3, 4, 5

    # Input constants
    NONE = 0
    NICKEL = 5
    DIME = 10

    def __init__(self):
        super().__init__(
            state_names=["IDLE", "FIVE", "TEN", "FIFTEEN", "DISPENSE", "CHANGE"],
            initial_state=self.IDLE,
        )
        self._transitions = {
            self.IDLE: {
                self.NONE: self.IDLE,
                self.NICKEL: self.FIVE,
                self.DIME: self.TEN,
            },
            self.FIVE: {
                self.NONE: self.FIVE,
                self.NICKEL: self.TEN,
                self.DIME: self.FIFTEEN,
            },
            self.TEN: {
                self.NONE: self.TEN,
                self.NICKEL: self.FIFTEEN,
                self.DIME: self.DISPENSE,
            },
            self.FIFTEEN: {
                self.NONE: self.FIFTEEN,
                self.NICKEL: self.DISPENSE,
                self.DIME: self.DISPENSE,
            },
            self.DISPENSE: {
                self.NONE: self.CHANGE,
                self.NICKEL: self.CHANGE,
                self.DIME: self.CHANGE,
            },
            self.CHANGE: {
                self.NONE: self.IDLE,
                self.NICKEL: self.IDLE,
                self.DIME: self.IDLE,
            },
        }

    def step(self, input_val=None):
        if input_val is None:
            input_val = self.NONE
        self.current_state = self._transitions[self.current_state][input_val]
        return self.current_state

    def get_legal_transitions(self):
        pairs = set()
        for state, trans in self._transitions.items():
            for next_state in trans.values():
                pairs.add((state, next_state))
        return pairs
