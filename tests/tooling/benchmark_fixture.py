"""Deterministic special-prime metadata for synthetic conversion records."""
def conversion_parameters(dimension):
    primes = [0x8000000004a0001, 0x800000000b80001, 0x800000000ee0001]
    roots = []
    for modulus in primes:
        for candidate in range(2, 100):
            root = pow(candidate, (modulus - 1) // (2 * dimension), modulus)
            if pow(root, dimension, modulus) == modulus - 1:
                roots.append(root)
                break
    return dict(parameter_setup_included=True, p_moduli=primes, p_roots=roots, p_policy="first_disjoint_60_bit_primes",
                down_input_policy="seeded_arbitrary_qp_evaluations")
