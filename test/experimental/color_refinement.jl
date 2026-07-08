# Brute-force reference: iterate (color, sorted multiset of neighbor colors) to a
# fixed point, then canonicalize by order of first appearance so it is directly
# comparable to `canonical_color_refinement`'s output. Only valid for undirected
# graphs, where `neighbors` captures both edge directions at once.
function _naive_stable_coloring(g::AbstractGraph, alpha::AbstractVector{<:Integer})
    n = nv(g)
    colour = collect(alpha)
    while true
        sigs = [(colour[v], Tuple(sort([colour[w] for w in neighbors(g, v)]))) for v in 1:n]
        seen = Dict{Tuple,Int}()
        newcolour = Vector{Int}(undef, n)
        for v in 1:n
            newcolour[v] = get!(seen, sigs[v], length(seen) + 1)
        end
        newcolour == colour && return colour
        colour = newcolour
    end
end

function _canonicalize(c::AbstractVector{<:Integer})
    remap = Dict{Int,Int}()
    return [get!(remap, x, length(remap) + 1) for x in c]
end

@testset "Color refinement" begin
    # Path graph: endpoints (degree 1) split from interior (degree 2), and the
    # interior further splits by distance, giving 3 stable classes (a palindrome).
    g = path_graph(5)
    c = canonical_color_refinement(g, ones(Int, nv(g)), [1])
    @test c[1] == c[5]
    @test c[2] == c[4]
    @test length(unique(c)) == 3

    # Vertex-transitive graphs stay monochromatic: refinement cannot distinguish
    # any vertex from another.
    for g in (cycle_graph(6), complete_graph(5), cycle_graph(4))
        c = canonical_color_refinement(g, ones(Int, nv(g)), [1])
        @test length(unique(c)) == 1
    end

    # Star graph: center (high degree) separates from the leaves.
    g = star_graph(5)
    c = canonical_color_refinement(g, ones(Int, nv(g)), [1])
    @test length(unique(c)) == 2
    @test count(==(c[1]), c) == 1 # the center is in a class of its own

    # The refinement is invariant under relabeling: isomorphic graphs yield the
    # same multiset of color classes.
    g1 = path_graph(6)
    σ = randperm(nv(g1))
    g2 = SimpleGraph(nv(g1))
    for e in edges(g1)
        add_edge!(g2, σ[src(e)], σ[dst(e)])
    end
    c1 = canonical_color_refinement(g1, ones(Int, nv(g1)), [1])
    c2 = canonical_color_refinement(g2, ones(Int, nv(g2)), [1])
    @test sort([count(==(x), c1) for x in unique(c1)]) ==
        sort([count(==(x), c2) for x in unique(c2)])

    # Directed graph: a directed path is fully discretized, since each vertex's
    # out-structure differs (the sink has out-degree 0, its predecessor's only
    # out-neighbor is the sink, and so on down the chain).
    dg = path_digraph(4)
    c = canonical_color_refinement(dg, ones(Int, nv(dg)), [1])
    @test c == [1, 2, 3, 4]

    # NOTE: refinement on digraphs only splits classes by out-edge structure (it
    # walks `inneighbors` to count, for each vertex, its out-edges into the color
    # class being processed); it never splits on in-edge structure. So two vertices
    # with identical out-neighbors but different in-neighbors are NOT distinguished.
    # This matches the reference algorithm, which is defined for undirected graphs;
    # this is documented current behavior for digraphs, not a guarantee of full
    # 1-WL correctness on them.
    dg2 = SimpleDiGraph(4)
    add_edge!(dg2, 1, 3) # 1 and 2 both point only to 3 ...
    add_edge!(dg2, 2, 3)
    add_edge!(dg2, 4, 1) # ... but 1 (unlike 2) also has an incoming edge from 4
    c = canonical_color_refinement(dg2, ones(Int, nv(dg2)), [1])
    @test c[1] == c[2]

    # The public experimental wrapper should be available and behave like the
    # canonical implementation.
    g_wrapper = path_graph(5)
    c_wrapper = canonical_color_refinement(g_wrapper, ones(Int, nv(g_wrapper)), [1])
    @test Graphs.Experimental.color_refinement(g_wrapper, ones(Int, nv(g_wrapper)), [1]) == c_wrapper

    # Convenience overloads should construct the default alpha and seed values.
    @test canonical_color_refinement(g_wrapper) == c_wrapper
    @test canonical_color_refinement(g_wrapper, ones(Int, nv(g_wrapper))) == c_wrapper
    @test canonical_color_refinement(g_wrapper, 1) == c_wrapper
    @test color_refinement(g_wrapper) == c_wrapper
    @test color_refinement(g_wrapper, ones(Int, nv(g_wrapper))) == c_wrapper
    @test color_refinement(g_wrapper, 1) == c_wrapper

    # Empty graphs should be handled without errors.
    @test canonical_color_refinement(SimpleGraph(0), Int[], Int[]) == Int[]

    # Arbitrary integer labels should be accepted and refined without requiring a
    # dense range of 1:nv(g). The output is canonical: it depends only on the
    # partition structure of `alpha`, not on the label values themselves, so
    # relabeling `alpha` (while keeping the same partition and seed) leaves the
    # result unchanged.
    c = canonical_color_refinement(path_graph(5), [100, 100, 200, 200, 100], [100])
    c_alt = canonical_color_refinement(path_graph(5), [7, 7, 9, 9, 7], [7])
    @test c == c_alt

    # Arbitrary integer labels, including zero and negative values, are accepted.
    # Vertex 2 (middle) has a neighbor in the seed class {1} that vertex 3 (leaf)
    # lacks, so refinement distinguishes all three vertices.
    c = canonical_color_refinement(path_graph(3), [0, 1, 1], [0])
    @test c[1] != c[2]
    @test c[2] != c[3]
    @test c[1] != c[3]

    # An initial coloring of the wrong length is rejected.
    @test_throws ArgumentError canonical_color_refinement(path_graph(3), ones(Int, 2), [1])

    # Self-loops are accepted; a self-loop makes vertex 1 structurally unique here.
    g_loop = SimpleGraph(3)
    add_edge!(g_loop, 1, 1)
    add_edge!(g_loop, 1, 2)
    add_edge!(g_loop, 2, 3)
    c = canonical_color_refinement(g_loop, ones(Int, 3), [1])
    @test length(unique(c)) == 3

    # Disconnected graphs: isomorphic components land on identical colors (color
    # refinement has no notion of "component id"), and an isolated vertex forms
    # its own class.
    g_disc = SimpleGraph(7)
    add_edge!(g_disc, 1, 2)
    add_edge!(g_disc, 2, 3)
    add_edge!(g_disc, 4, 5)
    add_edge!(g_disc, 5, 6)
    c = canonical_color_refinement(g_disc, ones(Int, 7), [1])
    @test c[1:3] == c[4:6]
    @test c[1] != c[2] # endpoint vs. middle of each path component
    @test c[7] ∉ c[1:6] # the isolated vertex is in a class of its own

    # Idempotence: re-refining an already-stable coloring, seeded with all of its
    # classes, changes nothing (up to the canonical relabeling).
    g_idem = path_graph(5)
    c = canonical_color_refinement(g_idem, ones(Int, nv(g_idem)), [1])
    @test canonical_color_refinement(g_idem, c, unique(c)) == c

    # Seed-choice invariance: seeding only the smaller of two initial classes is
    # sufficient to reach the same stable partition as seeding every class (the
    # classical equitable-refinement guarantee for the worklist algorithm).
    g_seed = path_graph(6)
    alpha_seed = [1, 1, 1, 1, 2, 2] # class 2 (size 2) is smaller than class 1 (size 4)
    c_small_seed = canonical_color_refinement(g_seed, alpha_seed, [2])
    c_all_seed = canonical_color_refinement(g_seed, alpha_seed, [1, 2])
    @test c_small_seed == c_all_seed

    # Multiple simultaneous seed colors are refined correctly together. Seeding
    # every initial class is always sufficient to reach the true stable partition
    # (unlike seeding an arbitrary strict subset of 3+ classes, which is not
    # guaranteed to fully stabilize).
    g_multi = path_graph(7)
    alpha_multi = [1, 1, 2, 2, 2, 3, 3]
    c = canonical_color_refinement(g_multi, alpha_multi, [1, 2, 3])
    @test length(unique(c)) == length(unique(_naive_stable_coloring(g_multi, alpha_multi)))

    # A known exact case: complete bipartite graph, sides distinguished by size.
    g_bip = complete_bipartite_graph(2, 3)
    c = canonical_color_refinement(g_bip, ones(Int, nv(g_bip)), [1])
    @test c == [1, 1, 2, 2, 2]

    # Property-based check against an independent brute-force fixed-point
    # computation, across randomized undirected graphs and initial colorings.
    rng = MersenneTwister(20260705)
    for _ in 1:30
        n = rand(rng, 3:10)
        p = rand(rng, (0.1, 0.3, 0.5, 0.7))
        g_rand = erdos_renyi(n, p; rng=rng)
        alpha_rand = rand(rng, 1:rand(rng, 1:min(4, n)), n)
        expected = _canonicalize(_naive_stable_coloring(g_rand, alpha_rand))
        actual = canonical_color_refinement(g_rand, alpha_rand, unique(alpha_rand))
        @test actual == expected
    end
end
