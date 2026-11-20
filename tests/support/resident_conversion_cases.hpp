#pragma once
// Actual OpenFHE public operations independently supply expected arrays.
// Included by the same host and native-CUDA resident test executable.
#include "ntt/resident.hpp"
#include "lattice/lat-hal.h"
#include "math/math-hal.h"
#include "math/nbtheory.h"
#include <array>

namespace resident_conversion_cases {
using namespace lbcrypto;
using namespace ntt::proposal;
using Params = std::shared_ptr<DCRTPoly::Params>;

struct Tables {
  std::vector<NativeInteger> inverses, preconditions;
  std::vector<std::vector<NativeInteger>> hats;
  std::vector<DoubleNativeInt> barrett;
  Tables(const Params &source, const Params &target) {
    BigInteger product(1);
    for (auto p : source->GetParams()) product *= BigInteger(p->GetModulus().ConvertToInt());
    for (auto p : source->GetParams()) {
      const auto q = p->GetModulus();
      auto hat = product / BigInteger(q.ConvertToInt());
      inverses.emplace_back(hat.Mod(BigInteger(q.ConvertToInt())).ConvertToInt());
      inverses.back() = inverses.back().ModInverse(q);
      preconditions.push_back(inverses.back().PrepModMulConst(q));
      hats.emplace_back();
      for (auto t : target->GetParams())
        hats.back().emplace_back(hat.Mod(BigInteger(t->GetModulus().ConvertToInt())).ConvertToInt());
    }
    const auto wide = BigInteger(1).LShiftEq(128);
    for (auto p : target->GetParams())
      barrett.push_back((wide / BigInteger(p->GetModulus().ConvertToInt())).ConvertToInt<DoubleNativeInt>());
  }
};
inline Params params(uint64_t n, const std::vector<Tower> &towers) {
  std::vector<NativeInteger> q, roots;
  for (auto t : towers) { q.emplace_back(t.q); roots.emplace_back(t.psi); }
  return std::make_shared<DCRTPoly::Params>(2*n, q, roots);
}
inline std::vector<uint64_t> flatten(const DCRTPoly &p) {
  std::vector<uint64_t> out;
  for (const auto &tower : p.GetAllElements())
    for (size_t i = 0; i < tower.GetLength(); ++i) out.push_back(tower[i].ConvertToInt());
  return out;
}
inline DCRTPoly fixture(const Params &p, unsigned pattern) {
  const auto n = p->GetRingDimension();
  DCRTPoly out(p, Format::COEFFICIENT, false);
  uint64_t state = 0x20112026;
  for (size_t t = 0; t < p->GetParams().size(); ++t) {
    const auto q = p->GetParams()[t]->GetModulus();
    NativeVector values(n, q);
    for (size_t i = 0; i < n; ++i) {
      state = state*6364136223846793005ULL+1;
      values[i] = NativeInteger(pattern == 0 ? 0 : pattern == 1 ? q.ConvertToInt()-1
                                  : pattern == 2 ? (i == n-1) : state%q.ConvertToInt());
    }
    NativePoly tower(p->GetParams()[t], Format::COEFFICIENT);
    tower.SetValues(std::move(values), Format::COEFFICIENT);
    out.SetElementAtIndex(t, std::move(tower));
  }
  return out;
}
inline void equal(std::span<const uint64_t> a, const DCRTPoly &expected,
                   const char *label, size_t &comparisons) {
  const auto b = flatten(expected);
  require(a.size() == b.size(), "resident conversion result extent");
  for (size_t i = 0; i < a.size(); ++i) {
    ++comparisons;
    if (a[i] != b[i]) throw std::runtime_error(std::string(label)+" at "+std::to_string(i));
  }
}
template<class Factory>
size_t run(Factory make_session, bool smoke) {
  size_t comparisons = 0;
  for (uint64_t n : {64ULL, 4096ULL, 65536ULL}) {
    if (smoke && n != 65536) continue;
    for (size_t q_count : {size_t(5), size_t(40)}) {
      if (q_count == 40 && (smoke || n != 65536)) continue;
      // Mixed widths and unsorted order catch incorrect tower/table selection.
      std::vector<Tower> full_q, p_towers;
      std::vector<uint64_t> used;
      const std::array<unsigned, 4> widths{55U, 30U, 60U, 50U};
      for (size_t i = 0; i < q_count+3; ++i) {
        const auto bits = widths[i%widths.size()];
        auto q = LastPrime<NativeInteger>(bits, 2*n);
        while (std::find(used.begin(), used.end(), q.ConvertToInt()) != used.end())
          q = PreviousPrime(q, 2*n);
        used.push_back(q.ConvertToInt());
        (full_q.size() < q_count ? full_q : p_towers).push_back(
            {q.ConvertToInt(), RootOfUnity(2*n, q).ConvertToInt()});
      }
      auto context = Parameters::create(n, full_q, p_towers, 3);
      const auto p = params(n, p_towers);
      const auto levels = q_count == 40 ? std::vector<size_t>{40} : std::vector<size_t>{5,4,3,2,1};
      for (size_t active : levels) {
        if (smoke && active != 4) continue;
        const auto q = params(n, {full_q.begin(), full_q.begin()+active});
        auto qp_towers = std::vector<Tower>(full_q.begin(), full_q.begin()+active);
        qp_towers.insert(qp_towers.end(), p_towers.begin(), p_towers.end());
        const auto qp = params(n, qp_towers);
        const Tables up_tables(q, p), down_tables(p, q);
        BigInteger product_p(1);
        for (auto t : p_towers) product_p *= BigInteger(t.q);
        std::vector<NativeInteger> pinv, pinv_precon;
        for (size_t i = 0; i < active; ++i) {
          const NativeInteger qi(full_q[i].q);
          pinv.emplace_back(product_p.Mod(BigInteger(full_q[i].q)).ConvertToInt());
          pinv.back() = pinv.back().ModInverse(qi);
          pinv_precon.push_back(pinv.back().PrepModMulConst(qi));
        }
        auto down = [&](const DCRTPoly &input) {
          return input.ApproxModDown(q, p, pinv, pinv_precon, down_tables.inverses,
              down_tables.preconditions, down_tables.hats, down_tables.barrett, {}, {}, NativeInteger(0), {});
        };
        const PolynomialLayout full{context, active, qp_basis(*context, active), Domain::Coefficient, Order::Natural};
        const PolynomialLayout source{context, active, q_prefix(*context, active), Domain::Coefficient, Order::Natural};
        const auto full_eval = transform_layout(full, false), source_eval = transform_layout(source, false);
        for (unsigned pattern = 0; pattern < 4; ++pattern) {
          if ((smoke || q_count == 40) && pattern != 3) continue;
          const auto original = fixture(q, pattern);
          auto expected_up = original;
          expected_up.ApproxModUp(q, p, qp, up_tables.inverses, up_tables.preconditions,
                                  up_tables.hats, up_tables.barrett);
          const auto expected_down = down(expected_up);
          auto session = make_session(full);
          auto input = session.upload(source, flatten(original));
          auto lifted = session.mod_up(input);
          auto result = session.mod_down(lifted);
          require(session.stats().uploads == 1 && session.stats().downloads == 0
                    && session.stats().parameter_uploads == 3, "unexpected conversion host roundtrip");
          equal(session.download(result).wait_values(), expected_down, "composed ModUp/ModDown", comparisons);
          equal(session.download(lifted).wait_values(), expected_up, "coefficient ModUp", comparisons);
          equal(session.download(input).wait_values(), original, "coefficient input immutable", comparisons);
          auto original_eval = original; original_eval.SetFormat(Format::EVALUATION);
          auto eval_input = session.upload(source_eval, flatten(original_eval));
          auto eval_up = session.mod_up(eval_input);
          equal(session.download(eval_up).wait_values(), expected_up, "evaluation ModUp", comparisons);
          equal(session.download(eval_input).wait_values(), original_eval, "evaluation input immutable", comparisons);

          // Arbitrary QP input checks ModDown independently of our ModUp.
          const auto arbitrary_coeff = fixture(qp, pattern);
          auto arbitrary = arbitrary_coeff; arbitrary.SetFormat(Format::EVALUATION);
          auto arbitrary_input = session.upload(full_eval, flatten(arbitrary));
          equal(session.download(session.mod_down(arbitrary_input)).wait_values(), down(arbitrary),
                "independent ModDown", comparisons);
          equal(session.download(arbitrary_input).wait_values(), arbitrary, "QP input immutable", comparisons);
          auto coeff_input = session.upload(full, flatten(arbitrary_coeff));
          auto p_view = session.subview(coeff_input, active, 3, active);
          // Use the actual P suffix's coefficients, not the separately seeded fixture.
          DCRTPoly actual_p(p, Format::COEFFICIENT, false);
          for (size_t i = 0; i < 3; ++i)
            actual_p.SetElementAtIndex(i, arbitrary_coeff.GetElementAtIndex(active+i));
          const auto correction = actual_p.ApproxSwitchCRTBasis(p, q, down_tables.inverses,
              down_tables.preconditions, down_tables.hats, down_tables.barrett);
          equal(session.download(session.switch_basis(p_view, source)).wait_values(), correction,
                "offset P-to-Q coefficient switch", comparisons);
          session.close();
        }
      }
    }
  }
  return comparisons;
}
} // namespace resident_conversion_cases
