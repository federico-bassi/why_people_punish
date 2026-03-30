using Random
using DataFrames
using Statistics
using CSV 

# ==========================================================================================================
# 1) Basic objects
# ==========================================================================================================
const C = 1
const D = 2
const P = 3
const ACTION_LABELS = Dict(C => "C", D => "D", P => "P")
const TYPE_LABELS = Dict(:grim => "grim", :punisher => "punisher", :defector => "defector")
const TREATMENT_PARAM = Dict("non-cooperative" => (b=4, c=3, alpha=3, beta=12, l=1, m=15),
                            "cooperative" => (b=2, c=1, alpha=1, beta=4, l=3, m=15))

function profile_row(a_self::Int, a_other::Int)
    return 3*a_self + a_other - 2
end

function stage_payoff(a1::Int, a2::Int; b=4, c=3, alpha=3, beta=12, l=1, m=15)
    if a1 == C && a2 == C
        return l*(b-c)+m
    elseif a1 == C && a2 == D
        return l*(-c)+m
    elseif a1 == C && a2 == P
        return l*(-c-beta)+m
    elseif a1 == D && a2 == C
        return l*(b)+m
    elseif a1 == D && a2 == D
        return l*(0.0)+m
    elseif a1 == D && a2 == P
        return l*(-beta)+m
    elseif a1 == P && a2 == C
        return l*(b-alpha)+m
    elseif a1 == P && a2 == D
        return l*(-alpha)+m
    elseif a1 == P && a2 == P
        return l*(-alpha-beta)+m
    else
        error("Unknown action profile")
    end
end

# ==========================================================================================================
# 2) Automata
# ==========================================================================================================
function automaton_grim()
    A = zeros(Int, 10, 2)

    # actions
    A[1,1] = C
    A[1,2] = D

    # From state 1
    A[2,1] = 1 
    A[3,1] = 2
    A[4,1] = 2

    # Fill all other rows defensively
    for r in 5:10
        A[r,1] = 2
    end

    # From state 2 (defect forever)
    for r in 2:10
        A[r,2] = 2
    end

    return A
end

function automaton_punisher()
    A = zeros(Int, 10, 3)

    # Actions
    A[1, 1] = C
    A[1, 2] = P
    A[1, 3] = D

    # Cooperative state
    # CC observed
    A[2,1] = 1
    # CD or CP observed
    A[3,1] = 2
    A[4,1] = 2
    for r in 5:10
        A[r, 1] = 3
    end
    
    # Punishment state
    A[8, 2] = 1
    for r in 2:10
        if r != 8
            A[r,2] = 3
        end
    end
    
    # Defection state
    A[5, 3] = 1
    for r in 2:10
        if r!= 5
            A[r, 3] = 3
        end
    end

    return A
end

function automaton_defector()
    A = ones(Int, 10, 1)
    A[1,1] = D
    return A
end


# ==========================================================================================================
# 3. Player object
# ==========================================================================================================
mutable struct Player
    id::Int
    type::Symbol
    automaton::Matrix{Int}
    state::Int
    payoff_since_update::Float64
    rounds_since_update::Int
    opponents_since_update::Vector{Symbol}
    realized_rounds_since_update::Vector{Int}
end

function reset_state!(p::Player)
    p.state=1
end

function current_action(p::Player)
    return p.automaton[1, p.state]
end

function transition!(p::Player, a_self::Int, a_other::Int)
    row = profile_row(a_self, a_other)
    next_state = p.automaton[row, p.state]
    p.state = next_state
end

function automaton_from_type(t::Symbol)
    if t == :grim
        return automaton_grim()
    elseif t == :punisher
        return automaton_punisher()
    elseif t == :defector
        return automaton_defector()
    else
        error("Unknown type")
    end
end

function make_shadow_player(type::Symbol)
    Player(-1, type, automaton_from_type(type), 1, 0.0, 0, Symbol[], Int[])
end

function total_payoff_against_sequence(self_type::Symbol,
                                      opponent_types::Vector{Symbol},
                                      realized_rounds::Vector{Int},
                                      rng::AbstractRNG;
                                      delta::Float64=0.7,
                                      b::Int=4,
                                      c::Int=3,
                                      alpha::Int=3,
                                      beta::Int=12,
                                      l::Int=1,
                                      m::Int=15)

    @assert length(opponent_types) == length(realized_rounds)

    total_payoff = 0.0
    total_rounds = 0

    for (opp_type, R) in zip(opponent_types, realized_rounds)
        p_self = make_shadow_player(self_type)
        p_opp  = make_shadow_player(opp_type)

        outcome = play_supergame!(p_self, p_opp, rng;
                                  delta=delta, b=b, c=c, alpha=alpha,
                                  beta=beta, l=l, m=m,
                                  forced_rounds=R)

        total_payoff += outcome.total1
        total_rounds += outcome.rounds
    end

    return total_payoff, total_rounds
end

# ==========================================================================================================
# 4. Supergame
# ==========================================================================================================
function play_supergame!(p1::Player, p2::Player, rng::AbstractRNG;
                        delta::Float64=0.7,
                        b::Int=4,
                        c::Int=3,
                        alpha::Int=3,
                        beta::Int=12,
                        l::Int=1,
                        m::Int=15,
                        forced_rounds::Union{Nothing, Int}=nothing)

    reset_state!(p1)
    reset_state!(p2)

    history = NamedTuple[]
    cum1 = 0.0
    cum2 = 0.0
    round = 0

    while true
        round += 1

        a1 = current_action(p1)
        a2 = current_action(p2)

        payoff1 = stage_payoff(a1, a2, b=b, c=c, alpha=alpha, beta=beta, l=l, m=m)
        payoff2 = stage_payoff(a2, a1, b=b, c=c, alpha=alpha, beta=beta, l=l, m=m)
        
        cum1 += payoff1
        cum2 += payoff2

        push!(history, (round=round, a1=a1, a2=a2, payoff1=payoff1, payoff2=payoff2, cum1=cum1, cum2=cum2))

        transition!(p1, a1, a2)
        transition!(p2, a2, a1)

        if forced_rounds == nothing
            if rand(rng) > delta
                break
            end
        else
            if round >= forced_rounds
                break
            end
        end
    end

    return(
        history = history,
        avg1 = cum1/round,
        avg2 = cum2/round,
        total1 = cum1,
        total2 = cum2,
        rounds = round
    )
end
# ==========================================================================================================
# 5. Matching
# ==========================================================================================================
function random_matching(ids::Vector{Int}, rng::AbstractRNG)
    shuffled = shuffle(rng, ids)
    return [(shuffled[i], shuffled[i+1]) for i in 1:2:length(shuffled)]
end

# ==========================================================================================================
# 6. Meta-strategies
# ==========================================================================================================
function maybe_update_one_player!(players::Vector{Player}, rng::AbstractRNG;
                                    punisher_bias::Float64=0.0,
                                    delta::Float64=0.7,
                                    b::Int=4,
                                    c::Int=3,
                                    alpha::Int=3,
                                    beta::Int=12,
                                    l::Int=1,
                                    m::Int=15)

    p = rand(rng, players)

    if p.rounds_since_update == 0
        return
    end

    realized_avg = p.payoff_since_update / p.rounds_since_update
    current_value = realized_avg + (p.type == :punisher ? punisher_bias : 0.0)

    candidate_types = Symbol[:grim, :punisher, :defector]
    candidate_types = filter(t -> t != p.type, candidate_types)

    alt_type_1 = candidate_types[1]
    alt_type_2 = candidate_types[2]

    rng_cf1 = copy(rng)
    alt_total_1, alt_round_1 = total_payoff_against_sequence(
        alt_type_1,
        p.opponents_since_update,
        p.realized_rounds_since_update,
        rng_cf1;
        delta=delta, b=b, c=c, alpha=alpha, beta=beta, l=l, m=m
    )
    alt_avg_1 = alt_total_1 / alt_round_1
    alt_value_1 = alt_avg_1 + (alt_type_1 == :punisher ? punisher_bias : 0.0)

    rng_cf2 = copy(rng)
    alt_total_2, alt_round_2 = total_payoff_against_sequence(
        alt_type_2,
        p.opponents_since_update,
        p.realized_rounds_since_update,
        rng_cf2;
        delta=delta, b=b, c=c, alpha=alpha, beta=beta, l=l, m=m
    )
    alt_avg_2 = alt_total_2 / alt_round_2
    alt_value_2 = alt_avg_2 + (alt_type_2 == :punisher ? punisher_bias : 0.0)

    best_type = p.type

    alt1_beats_current = alt_value_1 > current_value && !isapprox(alt_value_1, current_value)
    alt2_beats_current = alt_value_2 > current_value && !isapprox(alt_value_2, current_value)

    if alt1_beats_current && alt2_beats_current
        if isapprox(alt_value_1, alt_value_2)
            best_type = rand(rng, [alt_type_1, alt_type_2])
        elseif alt_value_1 > alt_value_2
            best_type = alt_type_1
        else
            best_type = alt_type_2
        end
    elseif alt1_beats_current
        best_type = alt_type_1
    elseif alt2_beats_current
        best_type = alt_type_2
    end

    if best_type != p.type
        p.type = best_type
        p.automaton = automaton_from_type(best_type)
        p.state = 1
    end

    p.payoff_since_update = 0.0
    p.rounds_since_update = 0
    empty!(p.opponents_since_update)
    empty!(p.realized_rounds_since_update)
end

# ==========================================================================================================
# 7. Group
# ==========================================================================================================
function initialise_players(; Nc::Int=2, Nd::Int=3, Np::Int=1, id_start::Int=1)    
    players = Player[]
    counter = id_start-1

    for i in 1:Nc
        counter+=1
        push!(players, Player(counter, :grim, automaton_grim(), 1, 0.0, 0, Symbol[], Int[]))
    end

    for i in 1:Nd
        counter+=1
        push!(players, Player(counter, :defector, automaton_defector(), 1, 0.0, 0, Symbol[], Int[]))
    end

    for i in 1:Np
        counter+=1
        push!(players, Player(counter, :punisher, automaton_punisher(), 1, 0.0, 0, Symbol[], Int[]))
    end

    return players
end

# ==========================================================================================================
# 8. Simulate
# ==========================================================================================================
function simulate(; Nc::Int=2, Nd::Int=3, Np::Int=1,
                T::Int=20, delta::Float64=0.7,
                b::Int=4,c::Int=3, alpha::Int=3, beta::Int=12, l::Int=1, m::Int=15,
                punisher_bias::Float64 = 0.0,
                seed::Int=1234,
                id_start::Int=1,
                session_id::Int,
                group_id::Int,
                treatment::String
                )

    rng = MersenneTwister(seed)
    players = initialise_players(Nc=Nc, Nd=Nd, Np=Np, id_start=id_start)
    start_type_map = Dict(p.id => TYPE_LABELS[p.type] for p in players)
    
    results = DataFrame(
        session_id = Int[],
        group_id = Int[],
        treatment = String[],
        Nc = Int[],
        Np = Int[],
        Nd = Int[],
        supergame = Int[],
        pair_id = Int[],
        round = Int[],
        player_id = Int[],
        partner_id = Int[],
        type_start = String[],
        type_current = String[],
        action = String[],
        opponent_action = String[],
        payoff = Float64[],
        cum_payoff = Float64[]
    )

    local_indices = collect(1:length(players))
    
    for t in 1:T
        pairs = random_matching(local_indices, rng)
        for (pair_idx, (i,j)) in enumerate(pairs)
            p1 = players[i]
            p2 = players[j]

            type_current_1 = TYPE_LABELS[p1.type]
            type_current_2 = TYPE_LABELS[p2.type]

            outcome = play_supergame!(p1, p2, rng; delta=delta, b=b, c=c, alpha=alpha, beta=beta, m=m, l=l)

            p1.payoff_since_update += outcome.total1
            p2.payoff_since_update += outcome.total2

            p1.rounds_since_update += outcome.rounds
            p2.rounds_since_update += outcome.rounds

            push!(p1.opponents_since_update, p2.type)
            push!(p2.opponents_since_update, p1.type)

            push!(p1.realized_rounds_since_update, outcome.rounds)
            push!(p2.realized_rounds_since_update, outcome.rounds)
            
            for h in outcome.history
                push!(results, (
                    session_id, group_id, treatment, Nc, Np, Nd,
                    t, pair_idx, h.round,
                    p1.id, p2.id,
                    start_type_map[p1.id], type_current_1,
                    ACTION_LABELS[h.a1], ACTION_LABELS[h.a2],
                    h.payoff1, h.cum1
                ))

                push!(results, (
                    session_id, group_id, treatment, Nc, Np, Nd,
                    t, pair_idx, h.round,
                    p2.id, p1.id,
                    start_type_map[p2.id], type_current_2,
                    ACTION_LABELS[h.a2], ACTION_LABELS[h.a1],
                    h.payoff2, h.cum2
                ))
            end
        end

        maybe_update_one_player!(players, rng; punisher_bias=punisher_bias, delta=delta, b=b, c=c, alpha=alpha, beta=beta, l=l, m=m)
    end
    return results, players
end

# ==========================================================================================================
# 10. Draw group composition
# ==========================================================================================================
function draw_group_composition(N::Int, Nc_low::Int, Nc_high::Int, Np_low::Int, Np_high::Int, rng::AbstractRNG)
    Nc = rand(rng, Nc_low:Nc_high)
    Np = rand(rng, Np_low:Np_high)
    Nd = N - Nc - Np

    return Nc, Np, Nd
end

# ==========================================================================================================
# MAIN
# ==========================================================================================================
function main()
    n_sessions = 12
    n_groups_per_session = 4
    N = 6
    Nc_low = 1
    Nc_high = 3
    Np_low = 0
    Np_high = 2
    num_supergames = 20
    delta = 0.7
    master_results = DataFrame()
    rng_session = MersenneTwister(1234)

    next_id_start = 1

    for session_id in 1:n_sessions
        for group_id in 1:n_groups_per_session
            treatment = group_id in (1,2) ? "cooperative" : "non-cooperative"
            pars = TREATMENT_PARAM[treatment]

            Nc, Np, Nd = draw_group_composition(N, Nc_low, Nc_high, Np_low, Np_high, rng_session)
            
            results, final_players = simulate(Nc=Nc, Np=Np, Nd=Nd,
                                            T=num_supergames, delta=delta,
                                            b=pars.b, c=pars.c, alpha=pars.alpha, beta=pars.beta, l=pars.l, m=pars.m,
                                            punisher_bias = 1.0,
                                            seed=1000 + 10*session_id + group_id,
                                            id_start=next_id_start,
                                            session_id=session_id,
                                            group_id=group_id,
                                            treatment=treatment)

            append!(master_results, results)
            next_id_start += N
        end
    end

    master_results = master_results[:, [
        :session_id,
        :group_id,
        :treatment,
        :Nc,
        :Np,
        :Nd,
        :supergame,
        :pair_id,
        :round,
        :player_id,
        :partner_id,
        :type_start,
        :type_current,
        :action,
        :opponent_action,
        :payoff,
        :cum_payoff
    ]]

    println(master_results)
    output = "/Users/federicobassi/Desktop/why_people_punish/simulated_data/prova_1.csv"
    CSV.write(output, master_results)

    return master_results
end

master_results = main()