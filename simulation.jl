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
    A[2,1] = 1   # CC -> stay cooperative
    A[3,1] = 2   # CD -> trigger
    A[4,1] = 2   # CP -> trigger

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

function automaton_punisher(length_p_phase::Int=3)
    n_states = 2 + length_p_phase
    coop_state = 1
    defect_state = n_states

    A = zeros(Int, 10, n_states)

    # Actions
    A[1, coop_state] = C
    for s in 2:(1+length_p_phase)
        A[1, s] = P
    end
    A[1, defect_state] = D

    # Cooperative state
    # Own action is C, relevant profiles: CC, CD, CP
    A[2, coop_state] = coop_state      # CC -> keep cooperating
    A[3, coop_state] = 2               # CD -> start punishment
    A[4, coop_state] = 2               # CP -> also treat punishment/non-coop as trigger

    for r in 5:10
        A[r, coop_state] = 2
    end

    
    # Punishment states
    for k in 1:length_p_phase
        s = 1 + k

        # own action is P, so relevant profiles are PC, PD, PP
        A[8, s] = coop_state

        if k < length_p_phase
            A[9, s]  = s + 1   # PD -> continue punishment
            A[10, s] = s + 1   # PP -> treat as continued conflict
        else
            A[9, s]  = defect_state   # punishment exhausted
            A[10, s] = defect_state
        end

        # Defensive fill for rows not expected from this state
        for r in 2:7
            A[r, s] = A[9, s]
        end
    end

    # Permanent defection state
    for r in 2:10
        A[r, defect_state] = defect_state
    end

    return A
end

function automaton_defector(length_d_phase::Int=3)
    defect_states = 1:length_d_phase
    coop_state = length_d_phase + 1
    permanent_defect_state = length_d_phase + 2
    n_states = length_d_phase + 2

    A = zeros(Int, 10, n_states)

    # -------------------------
    # Initialise the actions (first row)
    # -------------------------
    for s in defect_states
        A[1,s] = D
    end

    A[1, coop_state] = C
    A[1, permanent_defect_state] = D

    # -------------------------
    # Defection phase
    # -------------------------
    for s in defect_states
        A[5, s] = 1
        A[6, s] = 1

        if s < length_d_phase
            A[7, s] = s+1
        else
            A[7, s] = coop_state
        end

        for r in 2:4
            A[r,s] = 1
        end
        for r in 8:10
            A[r,s] = 1
        end
    end

    # -------------------------
    # Cooperative state
    # -------------------------
    A[2,coop_state] = coop_state
    A[3,coop_state] = permanent_defect_state
    A[4,coop_state] = permanent_defect_state

    for r in 5:10
        A[r, coop_state] = permanent_defect_state
    end

    # -------------------------
    # Permanent defection state
    # -------------------------
    for r in 2:10
        A[r, permanent_defect_state] = permanent_defect_state
    end

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
    recent_supergame_avgs::Vector{Float64}
    bad_experiences::Int
    current_length_d_phase::Int
    current_length_p_phase::Int
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
                        m::Int=15)

    reset_state!(p1)
    reset_state!(p2)

    total1 = 0.0
    total2 = 0.0
    rounds = 0

    while true
        rounds += 1

        a1 = current_action(p1)
        a2 = current_action(p2)

        total1 += stage_payoff(a1, a2, b=b, c=c, alpha=alpha, beta=beta, l=l, m=m)
        total2 += stage_payoff(a2, a1, b=b, c=c, alpha=alpha, beta=beta, l=l, m=m)

        transition!(p1, a1, a2)
        transition!(p2, a2, a1)

        if rand(rng) > delta
            break
        end
    end

    return(
        avg1 = total1/rounds,
        avg2 = total2/rounds,
        total1 = total1,
        total2 = total2,
        rounds =rounds
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
function maybe_update_strategy!(p::Player, dd_payoff::Float64)
    last_avg = p.recent_supergame_avgs[end]
    is_bad = last_avg < dd_payoff

    if !is_bad
        return
    end

    p.bad_experiences += 1

    # ---------------------
    # Punishing cooperators
    # ---------------------
    if p.type == :punisher
        p.current_length_p_phase = max(p.current_length_p_phase - 1, 0)

        if p.current_length_p_phase == 0
            p.type = :grim
            p.automaton = automaton_grim()
        else
            p.automaton = automaton_punisher(p.current_length_p_phase)
        end
        p.state = 1
        return
    end

    # ---------------------
    # Defectors
    # ---------------------
    if p.type == :defector
        p.current_length_d_phase = max(p.current_length_d_phase - 1, 0)

        if p.current_length_d_phase == 0
            p.type = :grim
            p.automaton = automaton_grim()
        else
            p.automaton = automaton_defector(p.current_length_d_phase)
        end
        p.state = 1
        return
    end

    # ---------------------
    # Non-punishing cooperators
    # ---------------------
    if p.type == :grim && p.bad_experiences >= 3
        p.type = :defector
        p.automaton = automaton_defector(1)
        p.state = 1
        return
    end
end

# ==========================================================================================================
# 7. Group
# ==========================================================================================================
function initialise_players(; Nc::Int=2, Nd::Int=3, Np::Int=1, length_p_phase::Int=3, length_d_phase::Int=6)    
    players = Player[]
    counter = 0

    for i in 1:Nc
        counter+=1
        push!(players, Player(counter, :grim, automaton_grim(), 1, Float64[], 0, 0, 0))
    end

    for i in 1:Nd
        counter+=1
        push!(players, Player(counter, :defector, automaton_defector(length_d_phase), 1, Float64[], 0,length_d_phase,0))
    end

    for i in 1:Np
        counter+=1
        push!(players, Player(counter, :punisher, automaton_punisher(length_p_phase), 1, Float64[], 0, 0, length_p_phase))
    end

    return players
end

# ==========================================================================================================
# 8. Simulate
# ==========================================================================================================
function simulate(; Nc::Int=2, Nd::Int=3, Np::Int=1,
                T::Int=20, delta::Float64=0.7,
                length_p_phase::Int=3,length_d_phase::Int=3,
                b::Int=4,c::Int=3, alpha::Int=3, beta::Int=12, l::Int=1, m::Int=15,
                seed::Int=1234
                )

    rng = MersenneTwister(seed)
    players = initialise_players(Nc=Nc, Nd=Nd, Np=Np, length_d_phase=length_d_phase, length_p_phase=length_p_phase)
    start_type_map = Dict(p.id => TYPE_LABELS[p.type] for p in players)
    
    results = DataFrame(
        supergame = Int[],
        pair_id = Int[],
        player_id = Int[],
        partner_id = Int[],
        type_start = String[],
        type_current = String[],
        avg_payoff = Float64[],
        tot_payoff = Float64[],
        rounds = Int[]
    )

    ids = [p.id for p in players]
    for t in 1:T
        pairs = random_matching(ids, rng)
        for (pair_idx, (i,j)) in enumerate(pairs)
            p1 = players[i]
            p2 = players[j]
            outcome = play_supergame!(p1, p2, rng; delta=delta, b=b, c=c, alpha=alpha, beta=beta, m=m, l=l)
            
            push!(p1.recent_supergame_avgs, outcome.avg1)
            push!(p2.recent_supergame_avgs, outcome.avg2)

            push!(results, (t, pair_idx, p1.id, p2.id, start_type_map[p1.id], TYPE_LABELS[p1.type], outcome.avg1, outcome.total1, outcome.rounds))
            push!(results, (t, pair_idx, p2.id, p1.id, start_type_map[p1.id], TYPE_LABELS[p2.type], outcome.avg2, outcome.total2, outcome.rounds))
        end

        dd_payoff = stage_payoff(D,D, b=b, c=c, alpha=alpha, beta=beta, l=l, m=m)
        for p in players
            maybe_update_strategy!(p, dd_payoff)
        end
    end
    return results, players
end

# ==========================================================================================================
# 19. Draw group composition
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
n_sessions = 12
N = 6
Nc_low = 1
Nc_high = 3
Np_low = 0
Np_high = 2
master_results = DataFrame()
rng_session = MersenneTwister(1234)

for session_id in 1:n_sessions
    Nc, Np, Nd = draw_group_composition(N, Nc_low, Nc_high, Np_low, Np_high, rng_session)
    results, final_players = simulate(Nc=Nc, Np=Np, Nd=Nd, seed=1000 + session_id)

    results.session_id .= session_id
    results.Nc .= Nc
    results.Np .= Np
    results.Nd .= Nd

    append!(master_results, results)
end

println(master_results)
output = "/Users/federicobassi/Desktop/why_people_punish/data.csv"
CSV.write(output, master_results)