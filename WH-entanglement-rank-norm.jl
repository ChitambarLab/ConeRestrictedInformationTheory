using Convex
using LinearAlgebra
using SCS
using DelimitedFiles
using Test

solver = () -> SCS.Optimizer(verbose=0)
println("Here we scan over α∈[-1,1] to generate our data.")
println("It will take a minute to initialize the solver.")
#First we define the linear program for ||ρ_α^{⊗2}||_{Ent_1}
function _DblCopyWHEnt1Norm(λ,d)
    if !(0<=λ && λ<=1)
        throw(DomainError(λ,"λ must be in [0,1]."))
    end
    v = Variable(4)
    objective = λ^2 * v[1] + λ*(1-λ)*(v[2]+v[3])+(1-λ)^2 *v[4]
    problem = maximize(objective)
    problem.constraints += [
        #Trace norm condition
        (d+1)^2 * v[1] + (d+1)*(d-1)*(v[2]+v[3]) + (d-1)^2 * v[4] == 4/(d^2),
        #PPT conditions
        sum(v) >= 0,
        (v[1]+v[3])*(d+1) >= (v[2]+v[4])*(d-1),
        (v[1]+v[2])*(d+1) >= (v[3]+v[4])*(d-1),
        v[1]*(d+1)^2 + v[4]*(1-d)^2 >= (v[2]+v[3])*(d-1)*(d+1),
        #Positivity conditions
        v[1] >= 0 , v[2] >= 0 , v[3] >= 0 , v[4] >= 0
    ]
    solve!(problem, solver)
    return problem.optval
end

function _DblCopyWHEnt1Normv2(λ,d)
    if !(0<=λ && λ<=1)
        throw(DomainError(λ,"λ must be in [0,1]."))
    end
    v = Variable(4)
    objective = v[1] + (2*λ-1)*(v[2]+v[3]-v[4] + 2*v[4]*λ)
    problem = maximize(objective)
    problem.constraints += [
        #Trace norm condition
        d^2*(d*(d*v[1]+v[2]+v[3])+v[4]) == 1
        #Positivity conditions
        v[1]+v[2]+v[3]+v[4] >= 0
        v[1]-v[2]+v[3]-v[4] >= 0
        v[1]+v[2]-v[3]-v[4] >= 0
        v[1]-v[2]-v[3]+v[4] >= 0
        #PPT conditions
        v[1] >= 0
        v[1] + d*v[2] >= 0
        v[1] + d*v[3] >= 0
        v[1] + d*(v[2]+v[3]) + d^2 * v[4] >= 0
    ]
    solve!(problem, solver)
    return problem.optval
end

#Next we define the function that returns the single copy cone value
function _SingleCopyEntkNormWH(α,d,k)
    if !(-1<=α && α<=1)
        throw(DomainError(α,"α must be in [-1,1]."))
    elseif !(k >= 0 && isinteger(k))
        throw(DomainError(k,"k must be a positive integer"))
    end

    k == 1 ? r = (1+min(α,0))/(d*(d-α)) : r = (1+abs(α))/(d*(d-α))
    return r
end

@testset "Generate Data" begin
    #For scanning over the data
    d = 3
    α_range = -1:0.01:1
    ctr = 0
    print_ctr = 0
    norm_results =  zeros(length(α_range),7)
    ent_results = zeros(length(α_range),7)
    for α in α_range
        ctr += 1
        print_ctr += 1
        if print_ctr == 10
            println("Now checking α = ", α)
            print_ctr = 0
        end
        λ = ((1+d)*(1-α))/(2*(d-α))

        #Norm results
        dbl_val = _DblCopyWHEnt1Norm(λ,d)
        single_copy_k1 = _SingleCopyEntkNormWH(α,d,1)
        single_copy_k2 = _SingleCopyEntkNormWH(α,d,2)
        k1sq = single_copy_k1^2
        k2sq = single_copy_k2^2
        norm_results[ctr,:] = [α,λ,dbl_val,k1sq,k2sq,dbl_val-k1sq,dbl_val-k2sq]

        #Entropic results
        dbl_ent = -log2(dbl_val) + 2*log2(d)
        ent1_sing = -log2(single_copy_k1) + log2(d)
        ent2_sing = -log2(single_copy_k2) + log2(d)
        ent_results[ctr,:] = [α,λ,dbl_ent,2*ent1_sing,2*ent2_sing,dbl_ent-2*ent1_sing,dbl_ent-2*ent2_sing]
    end

    header1 = ["α" "λ" "Two Copy Ent_1 Norm" "One Copy Ent_1 Norm Squared" "One Copy Ent_2 Norm Squared" "Ent_1 norm dif" "Ent_1 v Ent_2 dif"]
    data_to_save1 = vcat(header1,norm_results)
    writedlm("norm_results.csv", data_to_save1, ',')
    header2 = ["alpha" "lambda" "Two Werner States Ent 1" "Single Werner State Ent 1 Dbld" "Single Werner State Ent 2 Dbld" "Ent 1 Dif" "Ent_1 v Ent_2 dif"]
    data_to_save2 = vcat(header2,ent_results)
    writedlm("entropy_results.csv", data_to_save2, ',')

    #We know that ||𝔽 ⊗ 𝔽||_{Ent1} = 2/(d(d-1)) (Zhu and Hayashi, 2010)
    #It follows that ||ρ_{λ=0} \otimes ρ_{λ=0} ||_{Ent1} = 2/(d^3(d-1))
    #So assuming that the last entry is α=1 (λ=0), this test will pass
    @test isapprox(norm_results[length(α_range),3], 2/(d^3*(d-1)) , atol=1e-6)
end
