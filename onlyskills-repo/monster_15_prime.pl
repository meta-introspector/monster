% Monster 15-Prime Bit Prediction - Prolog
:- module(monster_15_prime, [
    predict_bit/3,
    predictor_bits/2,
    prime_weighted_vote/2
]).

% 15 Monster primes
monster_prime(0, 2).
monster_prime(1, 3).
monster_prime(2, 5).
monster_prime(3, 7).
monster_prime(4, 11).
monster_prime(5, 13).
monster_prime(6, 17).
monster_prime(7, 19).
monster_prime(8, 23).
monster_prime(9, 29).
monster_prime(10, 31).
monster_prime(11, 41).
monster_prime(12, 47).
monster_prime(13, 59).
monster_prime(14, 71).

% Get predictor bits for a given bit index
predictor_bits(BitIndex, Predictors) :-
    findall(PredictorBit-Prime,
        (monster_prime(I, Prime),
         Shift is 1 << I,
         PredictorBit is BitIndex xor (Prime * Shift)),
        Predictors).

% Prime-weighted vote
prime_weighted_vote(Predictors, Vote) :-
    findall(Weight,
        (member(PredictorBit-Prime, Predictors),
         BitValue is PredictorBit /\ 1,
         (BitValue = 1 -> Weight = Prime ; Weight is -Prime)),
        Weights),
    sum_list(Weights, Vote).

% Predict bit value
predict_bit(BitIndex, PredictedValue, Confidence) :-
    predictor_bits(BitIndex, Predictors),
    prime_weighted_vote(Predictors, Vote),
    (Vote > 0 -> PredictedValue = 1 ; PredictedValue = 0),
    
    % Calculate confidence
    findall(P, monster_prime(_, P), Primes),
    sum_list(Primes, TotalWeight),
    abs(Vote, AbsVote),
    Confidence is (AbsVote * 100) // TotalWeight.

% Query predictions for range
predict_range(Start, End, Predictions) :-
    findall([bit(Bit), predicted(Pred), confidence(Conf)],
        (between(Start, End, Bit),
         predict_bit(Bit, Pred, Conf)),
        Predictions).

% Find high-confidence predictions
high_confidence_predictions(MinConfidence, Predictions) :-
    findall([bit(Bit), predicted(Pred), confidence(Conf)],
        (between(0, 1000, Bit),
         predict_bit(Bit, Pred, Conf),
         Conf >= MinConfidence),
        Predictions).

% Verify prediction against actual bit
verify_prediction(BitIndex, ActualBit, Correct) :-
    predict_bit(BitIndex, PredictedBit, _),
    (PredictedBit = ActualBit -> Correct = true ; Correct = false).

% Statistics
prediction_stats(Stats) :-
    predict_range(0, 999, Predictions),
    length(Predictions, Total),
    findall(Conf,
        member([_, _, confidence(Conf)], Predictions),
        Confidences),
    sum_list(Confidences, SumConf),
    AvgConf is SumConf / Total,
    findall(1,
        member([_, predicted(1), _], Predictions),
        Ones),
    length(Ones, OneCount),
    Stats = [
        total(Total),
        ones(OneCount),
        avg_confidence(AvgConf)
    ].
