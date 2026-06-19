using LsqFit
using Test
import Aqua
import JET

@testset "QA" begin
    @testset "Aqua" begin
        Aqua.test_all(LsqFit)
    end

    @testset "JET" begin
        # Check that there are no undefined global references and undefined field accesses
        JET.test_package(LsqFit; target_modules = (LsqFit,), mode = :typo)

        # Analyze methods based on their declared signature
        JET.test_package(LsqFit; target_modules = (LsqFit,))
    end
end
