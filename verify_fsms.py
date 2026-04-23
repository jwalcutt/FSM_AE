"""
Phase 1 verification: run known input sequences through each FSM and confirm
output traces match hand-computed expected results. Also confirms all states
are reachable in each FSM.
"""

from fsm import TrafficLightFSM, BitPatternDetectorFSM, VendingMachineFSM


def verify_traffic_light():
    """Verify TrafficLightFSM against hand-traced transitions.

    Expected cycle: RED(0) -> RED_YELLOW(1) -> GREEN(2) -> YELLOW(3) -> RED(0) -> ...
    """
    fsm = TrafficLightFSM()
    print("=" * 60)
    print("FSM 1 -- Traffic Light Controller")
    print("=" * 60)

    # 8 ticks = two full cycles
    inputs = [None] * 8
    trace = fsm.run(inputs)
    expected = [0, 1, 2, 3, 0, 1, 2, 3, 0]
    #           R  RY G  Y  R  RY G  Y  R

    print(f"Input:    {inputs}")
    print(f"Trace:    {trace}")
    print(f"Expected: {expected}")
    print(f"Names:    {fsm.trace_names(trace)}")
    assert trace == expected, f"FAIL: {trace} != {expected}"
    print("PASS: Trace matches expected output")

    # All states reachable
    states_visited = set(trace)
    assert states_visited == set(range(fsm.num_states)), \
        f"FAIL: not all states reachable. Visited: {states_visited}"
    print(f"PASS: All {fsm.num_states} states reachable: {sorted(states_visited)}")

    # Legal transitions
    legal = fsm.get_legal_transitions()
    print(f"  Legal transitions: {sorted(legal)}")
    assert len(legal) == 4  # each state has exactly one successor
    print()


def verify_bit_pattern_detector():
    """Verify BitPatternDetectorFSM against hand-traced transitions.

    Hand trace for input [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]:
      S0 ->1-> S1 ->0-> S2 ->1-> S3 ->1-> S4 ->0-> S2 ->0-> S0 ->1-> S1 ->0-> S2 ->1-> S3 ->1-> S4
    """
    fsm = BitPatternDetectorFSM()
    print("=" * 60)
    print("FSM 2 -- Serial Bit-Pattern Detector ('1011')")
    print("=" * 60)

    # Test 1: two occurrences of "1011"
    inputs = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    trace = fsm.run(inputs)
    expected = [0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4]

    print(f"Input:    {inputs}")
    print(f"Trace:    {trace}")
    print(f"Expected: {expected}")
    print(f"Names:    {fsm.trace_names(trace)}")
    assert trace == expected, f"FAIL: {trace} != {expected}"
    print("PASS: Trace matches expected output")

    # Test 2: overlapping pattern -- "10110" should keep prefix "10"
    inputs2 = [1, 0, 1, 1, 0, 1, 1]
    trace2 = fsm.run(inputs2)
    # S0->1->S1->0->S2->1->S3->1->S4->0->S2->1->S3->1->S4
    expected2 = [0, 1, 2, 3, 4, 2, 3, 4]

    print(f"\nInput:    {inputs2}")
    print(f"Trace:    {trace2}")
    print(f"Expected: {expected2}")
    assert trace2 == expected2, f"FAIL: {trace2} != {expected2}"
    print("PASS: Overlapping pattern trace correct")

    # Test 3: all zeros -- should stay in S0
    inputs3 = [0, 0, 0, 0]
    trace3 = fsm.run(inputs3)
    expected3 = [0, 0, 0, 0, 0]
    assert trace3 == expected3, f"FAIL: {trace3} != {expected3}"
    print("PASS: All-zeros trace stays in S0")

    # Test 4: consecutive 1s -- should stay in S1
    inputs4 = [1, 1, 1, 1]
    trace4 = fsm.run(inputs4)
    expected4 = [0, 1, 1, 1, 1]
    assert trace4 == expected4, f"FAIL: {trace4} != {expected4}"
    print("PASS: Consecutive-1s trace stays in S1 after first")

    # All states reachable (across all tests)
    all_states = set(trace) | set(trace2)
    assert all_states == set(range(fsm.num_states)), \
        f"FAIL: not all states reachable. Visited: {all_states}"
    print(f"PASS: All {fsm.num_states} states reachable: {sorted(all_states)}")

    legal = fsm.get_legal_transitions()
    print(f"  Legal transitions ({len(legal)}): {sorted(legal)}")
    print()


def verify_vending_machine():
    """Verify VendingMachineFSM against hand-traced transitions.

    Test 1: nickel, nickel, dime -> 20c exact
      IDLE ->5-> FIVE ->5-> TEN ->10-> DISPENSE ->0-> CHANGE ->0-> IDLE

    Test 2: nickel, dime, nickel -> 20c exact (exercises FIFTEEN)
      IDLE ->5-> FIVE ->10-> FIFTEEN ->5-> DISPENSE ->0-> CHANGE ->0-> IDLE

    Test 3: dime, dime -> 20c exact
      IDLE ->10-> TEN ->10-> DISPENSE ->0-> CHANGE ->0-> IDLE
    """
    fsm = VendingMachineFSM()
    N, D = VendingMachineFSM.NICKEL, VendingMachineFSM.DIME
    print("=" * 60)
    print("FSM 3 -- Vending Machine Controller")
    print("=" * 60)

    # Test 1: nickel + nickel + dime = 20c
    inputs1 = [N, N, D, 0, 0]
    trace1 = fsm.run(inputs1)
    expected1 = [0, 1, 2, 4, 5, 0]
    #            IDLE FIVE TEN DISP CHNG IDLE

    print(f"Input:    {inputs1}")
    print(f"Trace:    {trace1}")
    print(f"Expected: {expected1}")
    print(f"Names:    {fsm.trace_names(trace1)}")
    assert trace1 == expected1, f"FAIL: {trace1} != {expected1}"
    print("PASS: Test 1 (N+N+D=20c) correct")

    # Test 2: nickel + dime + nickel = 20c (exercises FIFTEEN)
    inputs2 = [N, D, N, 0, 0]
    trace2 = fsm.run(inputs2)
    expected2 = [0, 1, 3, 4, 5, 0]
    #            IDLE FIVE FIFT DISP CHNG IDLE

    print(f"\nInput:    {inputs2}")
    print(f"Trace:    {trace2}")
    print(f"Expected: {expected2}")
    print(f"Names:    {fsm.trace_names(trace2)}")
    assert trace2 == expected2, f"FAIL: {trace2} != {expected2}"
    print("PASS: Test 2 (N+D+N=20c) correct -- FIFTEEN state reached")

    # Test 3: dime + dime = 20c
    inputs3 = [D, D, 0, 0]
    trace3 = fsm.run(inputs3)
    expected3 = [0, 2, 4, 5, 0]
    #            IDLE TEN DISP CHNG IDLE

    print(f"\nInput:    {inputs3}")
    print(f"Trace:    {trace3}")
    print(f"Expected: {expected3}")
    print(f"Names:    {fsm.trace_names(trace3)}")
    assert trace3 == expected3, f"FAIL: {trace3} != {expected3}"
    print("PASS: Test 3 (D+D=20c) correct")

    # Test 4: dime + nickel + nickel + nickel = 25c via FIFTEEN+dime path
    inputs4 = [N, D, D, 0, 0]
    trace4 = fsm.run(inputs4)
    expected4 = [0, 1, 3, 4, 5, 0]
    #            IDLE FIVE FIFT DISP CHNG IDLE (FIFTEEN+DIME=25c overpay)

    print(f"\nInput:    {inputs4}")
    print(f"Trace:    {trace4}")
    print(f"Expected: {expected4}")
    print(f"Names:    {fsm.trace_names(trace4)}")
    assert trace4 == expected4, f"FAIL: {trace4} != {expected4}"
    print("PASS: Test 4 (N+D+D=25c overpay) correct")

    # Test 5: no input -- stays IDLE
    inputs5 = [0, 0, 0]
    trace5 = fsm.run(inputs5)
    expected5 = [0, 0, 0, 0]
    assert trace5 == expected5, f"FAIL: {trace5} != {expected5}"
    print("PASS: Test 5 (no coins) stays IDLE")

    # All states reachable (across tests)
    all_states = set(trace1) | set(trace2) | set(trace3)
    assert all_states == set(range(fsm.num_states)), \
        f"FAIL: not all states reachable. Visited: {all_states}"
    print(f"PASS: All {fsm.num_states} states reachable: {sorted(all_states)}")

    legal = fsm.get_legal_transitions()
    print(f"  Legal transitions ({len(legal)}): {sorted(legal)}")
    print()


if __name__ == "__main__":
    verify_traffic_light()
    verify_bit_pattern_detector()
    verify_vending_machine()
    print("=" * 60)
    print("ALL PHASE 1 VERIFICATIONS PASSED")
    print("=" * 60)
