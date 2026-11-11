// D8: pinned OpenFHE Hybrid graph against unreduced-CRT and scalar MAC oracles.
// Every polynomial has N=65536. No CUDA or approximate-decryption result.
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "openfhe.h"
#include "keyswitch/keyswitch-hybrid.h"
#include "ntt/ntt.hpp"

namespace {
using namespace lbcrypto;
using Rows = std::vector<std::vector<uint64_t>>;
using Params = std::shared_ptr<DCRTPoly::Params>;
constexpr size_t N = 65536;
uint64_t comparisons = 0, noncanonical_witnesses = 0, wrong_key_witnesses = 0;

void require(bool ok, const std::string& where) {
  if (!ok) throw std::runtime_error(where);
}
uint64_t modulus(const Params& p, size_t i) { return p->GetParams().at(i)->GetModulus().ConvertToInt(); }
uint64_t root(const Params& p, size_t i) { return p->GetParams().at(i)->GetRootOfUnity().ConvertToInt(); }
Params select(const Params& p, size_t first, size_t count) {
  std::vector<NativeInteger> qs, roots;
  for (size_t i = first; i < first + count; ++i) {
    qs.push_back(p->GetParams().at(i)->GetModulus());
    roots.push_back(p->GetParams().at(i)->GetRootOfUnity());
  }
  return std::make_shared<DCRTPoly::Params>(2 * N, qs, roots);
}
DCRTPoly make(const Rows& rows, const Params& params, Format format) {
  require(rows.size() == params->GetParams().size(), "fixture tower count");
  DCRTPoly result(params, format, false);
  for (size_t l = 0; l < rows.size(); ++l) {
    NativePoly tower(params->GetParams()[l], format);
    NativeVector data(N, NativeInteger(modulus(params, l)));
    for (size_t i = 0; i < N; ++i) data[i] = NativeInteger(rows[l][i]);
    tower.SetValues(std::move(data), format);
    result.SetElementAtIndex(l, std::move(tower));
  }
  return result;
}
void equal(const DCRTPoly& actual, const Rows& expected, const Params& params,
           Format format, const std::string& where) {
  require(actual.GetFormat() == format, where + ": outer format");
  require(actual.GetNumOfElements() == expected.size(), where + ": tower count");
  for (size_t l = 0; l < expected.size(); ++l) {
    const auto& tower = actual.GetElementAtIndex(l);
    require(tower.GetFormat() == format, where + ": child format");
    require(tower.GetModulus().ConvertToInt() == modulus(params, l), where + ": modulus order");
    require(tower.GetRootOfUnity().ConvertToInt() == root(params, l), where + ": root order");
    for (size_t i = 0; i < N; ++i) {
      const auto v = tower[i].ConvertToInt();
      if (v != expected[l][i] || v >= modulus(params, l))
        throw std::runtime_error(where + ": tower " + std::to_string(l) + ", index " + std::to_string(i));
    }
    comparisons += N;
  }
}
Rows transform(Rows data, const Params& params, bool inverse = false) {
  for (size_t l = 0; l < data.size(); ++l) {
    ntt::NegacyclicNtt plan(N, modulus(params, l), root(params, l));
    if (inverse) plan.inverse(data[l]); else plan.forward(data[l]);
  }
  return data;
}
// This intentionally does NOT reduce the CRT sum modulo its source product.
// No OpenFHE basis-switch tables or basis-switch routines are used here.
Rows lift(const Rows& source, const Params& from, const Params& to) {
  BigInteger product(1);
  for (size_t l = 0; l < source.size(); ++l) product *= BigInteger(modulus(from, l));
  std::vector<BigInteger> hats;
  std::vector<uint64_t> inverses;
  for (size_t l = 0; l < source.size(); ++l) {
    hats.push_back(product / BigInteger(modulus(from, l)));
    inverses.push_back(ntt::inv_mod(hats.back().Mod(BigInteger(modulus(from, l))).ConvertToInt(), modulus(from, l)));
  }
  Rows result(to->GetParams().size(), std::vector<uint64_t>(N));
  for (size_t i = 0; i < N; ++i) {
    BigInteger sum(0);
    for (size_t l = 0; l < source.size(); ++l)
      sum += hats[l] * BigInteger(ntt::mul_mod(source[l][i], inverses[l], modulus(from, l)));
    for (size_t l = 0; l < result.size(); ++l) result[l][i] = sum.Mod(BigInteger(modulus(to, l))).ConvertToInt();
    if (i < 32 && result.size() && sum >= product &&
        result[0][i] != sum.Mod(product).Mod(BigInteger(modulus(to, 0))).ConvertToInt())
      ++noncanonical_witnesses;
  }
  return result;
}
uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
Rows fixture(const Params& params, unsigned pattern) {
  Rows out(params->GetParams().size(), std::vector<uint64_t>(N));
  for (size_t l = 0; l < out.size(); ++l) for (size_t i = 0; i < N; ++i) {
    const auto q = modulus(params, l);
    if (pattern == 0) out[l][i] = 0;
    else if (pattern == 1) out[l][i] = q - 1;
    else if (pattern == 2) out[l][i] = i == 17 ? 1 : 0;
    else out[l][i] = i < 3 ? (i == 0 ? 0 : i == 1 ? 1 : q - 1) : mix(i + 12345 * (l + 1)) % q;
  }
  return out;
}
EvalKey<DCRTPoly> diagnostic_key(const CryptoContext<DCRTPoly>& cc,
                               const std::shared_ptr<CryptoParametersRNS>& cp) {
  const auto p = cp->GetParamsQP();
  std::vector<DCRTPoly> a, b;
  for (size_t digit = 0; digit < cp->GetNumPartQ(); ++digit) {
    Rows av(p->GetParams().size(), std::vector<uint64_t>(N)), bv = av;
    for (size_t l = 0; l < av.size(); ++l) for (size_t i = 0; i < N; ++i) {
      const auto q = modulus(p, l);
      av[l][i] = mix(0xaaa + i + 1000003 * l + 1000033 * digit) % q;
      bv[l][i] = mix(0xbbb + i + 1000037 * l + 1000039 * digit) % q;
    }
    a.push_back(make(av, p, Format::EVALUATION)); b.push_back(make(bv, p, Format::EVALUATION));
  }
  auto key = std::make_shared<EvalKeyRelinImpl<DCRTPoly>>(cc);
  key->SetAVector(std::move(a));key->SetBVector(std::move(b));
  return key; // deterministic diagnostic arrays, NOT a cryptographic evaluation key
}

std::vector<Rows> check_digits(const DCRTPoly& input, const Rows& coeff,
    const std::shared_ptr<CryptoParametersRNS>& cp,
    const std::shared_ptr<std::vector<DCRTPoly>>& actual) {
  const auto ql = input.GetParams(), p = cp->GetParamsP(), ext = input.GetExtendedCRTBasis(p);
  const size_t L = coeff.size(), K = p->GetParams().size(), alpha = cp->GetNumPerPartQ();
  const size_t beta = (L + alpha - 1) / alpha;
  require(actual->size() == beta, "active digit count");
  std::vector<Rows> expected;
  for (size_t digit = 0; digit < beta; ++digit) {
    const size_t start = digit * alpha, length = std::min(alpha, L - start);
    const auto part_params = select(ql, start, length);
    const auto complement = cp->GetParamsComplPartQ(L - 1, digit);
    Rows part_coeff(coeff.begin() + start, coeff.begin() + start + length);
    Rows part_eval = transform(part_coeff, part_params);
    auto part = make(part_eval, part_params, Format::EVALUATION);
    part.SetFormat(Format::COEFFICIENT);
    equal(part, part_coeff, part_params, Format::COEFFICIENT, "digit inverse NTT");
    const Rows complement_coeff = lift(part_coeff, part_params, complement);
    auto switched = part.ApproxSwitchCRTBasis(cp->GetParamsPartQ(digit), complement,
        cp->GetPartQlHatInvModq(digit, length - 1), cp->GetPartQlHatInvModqPrecon(digit, length - 1),
        cp->GetPartQlHatModp(L - 1, digit), cp->GetmodComplPartqBarrettMu(L - 1, digit));
    equal(switched, complement_coeff, complement, Format::COEFFICIENT, "digit approximate lift");
    const Rows complement_eval = transform(complement_coeff, complement);
    switched.SetFormat(Format::EVALUATION);
    equal(switched, complement_eval, complement, Format::EVALUATION, "complement forward NTT");
    Rows combined(L + K);
    for (size_t l = 0; l < L + K; ++l) {
      if (l >= start && l < start + length) combined[l] = part_eval[l - start];
      else combined[l] = complement_eval[l < start ? l : l - length];
    }
    equal(actual->at(digit), combined, ext, Format::EVALUATION, "public digit assembly");
    expected.push_back(std::move(combined));
  }
  return expected;
}
std::vector<Rows> check_mac(const std::vector<Rows>& digits, const EvalKey<DCRTPoly>& key,
                          const Params& active, const std::vector<DCRTPoly>& actual) {
  const auto cp = std::dynamic_pointer_cast<CryptoParametersRNS>(key->GetCryptoParameters());
  const size_t M = cp->GetElementParams()->GetParams().size(), L = active->GetParams().size();
  const auto ext = actual.at(0).GetParams();
  std::vector<Rows> out(2, Rows(ext->GetParams().size(), std::vector<uint64_t>(N)));
  require(actual.size() == 2, "MAC returns [b contribution, a contribution]");
  for (size_t channel = 0; channel < 2; ++channel) {
    const auto& keys = channel == 0 ? key->GetBVector() : key->GetAVector();
    for (size_t l = 0; l < out[channel].size(); ++l) {
      const size_t key_l = l < L ? l : l + M - L;
      const auto q = modulus(ext, l);
      for (size_t i = 0; i < N; ++i) {
        uint64_t wrong = 0;
        for (size_t d = 0; d < digits.size(); ++d) {
          const auto k = keys.at(d).GetElementAtIndex(key_l)[i].ConvertToInt();
          out[channel][l][i] = ntt::add_mod(out[channel][l][i], ntt::mul_mod(digits[d][l][i], k, q), q);
          if (L != M && l >= L) wrong = ntt::add_mod(wrong, ntt::mul_mod(digits[d][l][i],
              keys.at(d).GetElementAtIndex(l)[i].ConvertToInt() % q, q), q);
        }
        if (L != M && l >= L && i < 32 && wrong != out[channel][l][i]) ++wrong_key_witnesses;
      }
    }
    equal(actual[channel], out[channel], ext, Format::EVALUATION, "public QlP MAC and key offset");
  }
  return out;
}
Rows check_down(const DCRTPoly& actual_ext, const Rows& ext_eval, const Params& active,
                const std::shared_ptr<CryptoParametersRNS>& cp) {
  const size_t L = active->GetParams().size();
  const auto p = cp->GetParamsP();
  Rows p_eval(ext_eval.begin() + L, ext_eval.end());
  const Rows p_coeff = transform(p_eval, p, true);
  auto part = make(p_eval, p, Format::EVALUATION);part.SetFormat(Format::COEFFICIENT);
  equal(part, p_coeff, p, Format::COEFFICIENT, "ModDown inverse P NTT");
  const Rows correction_coeff = lift(p_coeff, p, active);
  auto correction = part.ApproxSwitchCRTBasis(p, active, cp->GetPHatInvModp(), cp->GetPHatInvModpPrecon(),
                                             cp->GetPHatModq(), cp->GetModqBarrettMu());
  equal(correction, correction_coeff, active, Format::COEFFICIENT, "ModDown approximate P-to-Q correction");
  const Rows correction_eval = transform(correction_coeff, active);
  correction.SetFormat(Format::EVALUATION);
  equal(correction, correction_eval, active, Format::EVALUATION, "ModDown forward Q NTT");
  uint64_t p_mod_q = 1;
  Rows expected(L, std::vector<uint64_t>(N));
  for (size_t l = 0; l < L; ++l) {
    const auto q = modulus(active, l);p_mod_q = 1;
    for (size_t j = 0; j < p->GetParams().size(); ++j) p_mod_q = ntt::mul_mod(p_mod_q, modulus(p, j) % q, q);
    const auto pinv = ntt::inv_mod(p_mod_q, q);
    require(pinv == cp->GetPInvModq().at(l).ConvertToInt(), "P inverse parameter");
    for (size_t i = 0; i < N; ++i)
      expected[l][i] = ntt::mul_mod(ntt::sub_mod(ext_eval[l][i], correction_eval[l][i], q), pinv, q);
  }
  const auto actual = actual_ext.ApproxModDown(active, p, cp->GetPInvModq(), cp->GetPInvModqPrecon(),
      cp->GetPHatInvModp(), cp->GetPHatInvModpPrecon(), cp->GetPHatModq(), cp->GetModqBarrettMu(),
      cp->GettInvModp(), cp->GettInvModpPrecon(), NativeInteger(0), cp->GettModqPrecon());
  equal(actual, expected, active, Format::EVALUATION, "public ApproxModDown");
  return expected;
}

void check_wrappers(const CryptoContext<DCRTPoly>& cc, const KeySwitchHYBRID& ks,
                    const EvalKey<DCRTPoly>& key, const DCRTPoly& input, const Rows& base,
                    const std::vector<Rows>& down) {
  const auto params = input.GetParams();
  for (size_t count : {size_t(2), size_t(3)}) {
    auto ct = std::make_shared<CiphertextImpl<DCRTPoly>>(cc);
    auto initial = make(base, params, Format::COEFFICIENT);
    std::vector<DCRTPoly> components{initial};
    if (count == 3) components.push_back(initial);
    components.push_back(input);ct->SetElements(std::move(components));
    const Rows base_eval = transform(base, params);
    ks.KeySwitchInPlace(ct, key);
    require(ct->GetElements().size() == 2, "outer wrapper trims to two components");
    auto expected = down;
    for (size_t l = 0; l < base.size(); ++l) for (size_t i = 0; i < N; ++i) {
      expected[0][l][i] = ntt::add_mod(expected[0][l][i], base_eval[l][i], modulus(params, l));
      if (count == 3) expected[1][l][i] = ntt::add_mod(expected[1][l][i], base_eval[l][i], modulus(params, l));
    }
    for (size_t channel = 0; channel < 2; ++channel)
      equal(ct->GetElements()[channel], expected[channel], params, Format::EVALUATION, "outer add/replace semantics");
  }
}
// Extended ciphertext scaling is a separate operation from digit basis extension.
void check_delayed(const CryptoContext<DCRTPoly>& cc, const KeySwitchHYBRID& ks,
                   const DCRTPoly& input, const Rows& evaluation,
                   const std::vector<DCRTPoly>& mac, const std::vector<Rows>& down,
                   const std::shared_ptr<CryptoParametersRNS>& cp) {
  const auto active = input.GetParams(), ext_params = mac[0].GetParams();
  const size_t L = evaluation.size(), total = ext_params->GetParams().size();
  for (bool add_first : {false, true}) {
    auto ct = std::make_shared<CiphertextImpl<DCRTPoly>>(cc);
    ct->SetElements(std::vector<DCRTPoly>{input, input});
    auto extended = ks.KeySwitchExt(ct, add_first);
    std::vector<Rows> expected(2, Rows(total, std::vector<uint64_t>(N)));
    for (size_t ch = 0; ch < 2; ++ch) {
      if (add_first || ch != 0) for (size_t l = 0; l < L; ++l)
        for (size_t i = 0; i < N; ++i)
          expected[ch][l][i] = ntt::mul_mod(evaluation[l][i], cp->GetPModq()[l].ConvertToInt(), modulus(active,l));
      equal(extended->GetElements()[ch], expected[ch], ext_params, Format::EVALUATION,
            "KeySwitchExt scales Q and appends zero P towers");
    }
    const auto recovered = ks.KeySwitchDown(extended);
    for (size_t ch = 0; ch < 2; ++ch) {
      const Rows want = add_first || ch != 0 ? evaluation : Rows(L, std::vector<uint64_t>(N));
      equal(recovered->GetElements()[ch], want, active, Format::EVALUATION, "Down(Ext) identity");
    }
    for (size_t ch = 0; ch < 2; ++ch) extended->GetElements()[ch] += mac[ch];
    const auto combined = ks.KeySwitchDown(extended);
    auto combined_expected = down;
    for (size_t ch = 0; ch < 2; ++ch) {
      if (add_first || ch != 0) for (size_t l = 0; l < L; ++l)
        for (size_t i = 0; i < N; ++i)
          combined_expected[ch][l][i] = ntt::add_mod(down[ch][l][i], evaluation[l][i], modulus(active,l));
      equal(combined->GetElements()[ch], combined_expected[ch], active, Format::EVALUATION,
            "delayed QP accumulation and KeySwitchDown");
    }
    equal(ks.KeySwitchDownFirstElement(extended), combined_expected[0], active, Format::EVALUATION,
          "first-element-only delayed reduction");
  }
}
void check_reduction_order(const std::vector<Rows>& digits, const EvalKey<DCRTPoly>& key,
                           const Params& active, const std::vector<Rows>& correct,
                           const std::shared_ptr<CryptoParametersRNS>& cp) {
  const size_t L = active->GetParams().size(), M = cp->GetElementParams()->GetParams().size();
  auto active_zero = make(Rows(L, std::vector<uint64_t>(N)), active, Format::EVALUATION);
  const auto active_ext = active_zero.GetExtendedCRTBasis(cp->GetParamsP());
  uint64_t witnesses = 0;
  for (size_t ch = 0; ch < 2; ++ch) {
    const auto& keys = ch == 0 ? key->GetBVector() : key->GetAVector();
    Rows wrong(L, std::vector<uint64_t>(N));
    for (size_t d = 0; d < digits.size(); ++d) {
      Rows product(digits[d].size(), std::vector<uint64_t>(N));
      for (size_t l = 0; l < product.size(); ++l) for (size_t i = 0; i < N; ++i)
        product[l][i] = ntt::mul_mod(digits[d][l][i], keys[d].GetElementAtIndex(l < L ? l : l + M - L)[i].ConvertToInt(), modulus(active_ext,l));
      auto term = make(product, active_ext, Format::EVALUATION);
      const auto reduced = term.ApproxModDown(active, cp->GetParamsP(), cp->GetPInvModq(), cp->GetPInvModqPrecon(),
          cp->GetPHatInvModp(), cp->GetPHatInvModpPrecon(), cp->GetPHatModq(), cp->GetModqBarrettMu(),
          cp->GettInvModp(), cp->GettInvModpPrecon(), NativeInteger(0), cp->GettModqPrecon());
      for (size_t l = 0; l < L; ++l) for (size_t i = 0; i < N; ++i)
        wrong[l][i] = ntt::add_mod(wrong[l][i], reduced.GetElementAtIndex(l)[i].ConvertToInt(), modulus(active,l));
    }
    for (size_t l = 0; l < L; ++l) for (size_t i = 0; i < N; ++i) witnesses += wrong[l][i] != correct[ch][l][i];
  }
  require(witnesses > 0, "fixture must reject moving ModDown inside the digit MAC");
  std::cout << "REDUCTION ORDER OK: " << witnesses << " residues distinguish sum-then-ModDown from ModDown-then-sum" << std::endl;
}
void run_case(const CryptoContext<DCRTPoly>& cc, const KeySwitchHYBRID& ks,
              const std::shared_ptr<CryptoParametersRNS>& cp, const EvalKey<DCRTPoly>& key,
              size_t L, unsigned pattern, bool wrappers) {
  const auto active = select(cp->GetElementParams(), 0, L);
  const Rows coeff = fixture(active, pattern), evaluation = transform(coeff, active);
  const auto input = make(evaluation, active, Format::EVALUATION);
  const auto digits = ks.EvalKeySwitchPrecomputeCore(input, cp);
  const auto expected_digits = check_digits(input, coeff, cp, digits);
  const auto ext = ks.EvalFastKeySwitchCoreExt(digits, key, active);
  const auto expected_ext = check_mac(expected_digits, key, active, ext);
  const auto fast = ks.EvalFastKeySwitchCore(digits, key, active);
  const auto full = ks.KeySwitchCore(input, key);
  std::vector<Rows> down;
  for (size_t channel = 0; channel < 2; ++channel) {
    down.push_back(check_down(ext[channel], expected_ext[channel], active, cp));
    equal(fast[channel], down.back(), active, Format::EVALUATION, "fast core composition");
    equal(full[channel], down.back(), active, Format::EVALUATION, "full core composition");
  }
  if (wrappers) check_wrappers(cc, ks, key, input, coeff, down);
  if (wrappers && L == 4 && pattern == 3) {
    check_delayed(cc, ks, input, evaluation, ext, down, cp);
    check_reduction_order(expected_digits, key, active, down, cp);
  }
  equal(input, evaluation, active, Format::EVALUATION, "source input remains unchanged");
  std::cout << "HYBRID GRAPH OK: N=" << N << " L=" << L << " pattern=" << pattern
            << " digits=" << digits->size() << " key=" << (wrappers ? "diagnostic+wrapper" : "generated") << std::endl;
}
} // namespace

int main() {
  try {
    CCParams<CryptoContextCKKSRNS> config;
    config.SetRingDim(N);config.SetMultiplicativeDepth(4);config.SetScalingModSize(50);
    config.SetFirstModSize(60);config.SetScalingTechnique(FIXEDMANUAL);
    config.SetKeySwitchTechnique(HYBRID);config.SetNumLargeDigits(2);
    auto cc = GenCryptoContext(config);cc->Enable(PKE);cc->Enable(KEYSWITCH);cc->Enable(LEVELEDSHE);
    auto cp = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());
    require(cc->GetRingDimension() == N, "fixed full ring size");
    require(cp->GetElementParams()->GetParams().size() == 5 && cp->GetNumPerPartQ() == 3,
            "fixture requires five Q towers, alpha=3");
    require(cp->GetParamsP()->GetParams().size() == 3 && cp->GetNoiseScale() == 1, "CKKS three-P/no-t branch");
    const std::array<uint64_t,8> fixed_q{1152921504606584833ULL,1125899902124033ULL,1125899911168001ULL,
      1125899903827969ULL,1125899908022273ULL,1152921504598720513ULL,1152921504597016577ULL,1152921504595968001ULL};
    const std::array<uint64_t,8> fixed_psi{18043022392882ULL,15995550378ULL,21670117260ULL,938640682ULL,
      6511788946ULL,800790938143ULL,17749908910371ULL,11469071954203ULL};
    for (size_t l=0;l<8;++l) require(modulus(cp->GetParamsQP(),l)==fixed_q[l] && root(cp->GetParamsQP(),l)==fixed_psi[l],
                                   "pinned deterministic fixture parameters");
    std::cout << "CONTEXT N=" << N << " M=5 alpha=3 beta=2 K=" << cp->GetParamsP()->GetParams().size() << std::endl;
    for (size_t l = 0; l < cp->GetParamsQP()->GetParams().size(); ++l)
      std::cout << "PARAM " << l << " q=" << modulus(cp->GetParamsQP(), l)
                << " psi=" << root(cp->GetParamsQP(), l) << std::endl;
    KeySwitchHYBRID ks;
    const auto diagnostic = diagnostic_key(cc, cp);
    for (size_t L : {size_t(5), size_t(4), size_t(3), size_t(1)})
      for (unsigned pattern = 0; pattern < 4; ++pattern)
        run_case(cc, ks, cp, diagnostic, L, pattern, true);
    require(noncanonical_witnesses > 0, "fixtures must detect canonical-CRT substitution");
    require(wrong_key_witnesses > 0, "fixtures must detect missing reduced-level P offset");
    // Reuse one actual generated key inside each exact array comparison. Its randomness
    // is not compared across runs; deterministic diagnostic fixtures above are repeatable.
    const auto old_keys = cc->KeyGen(), new_keys = cc->KeyGen();
    const auto generated = cc->KeySwitchGen(old_keys.secretKey, new_keys.secretKey);
    run_case(cc, ks, cp, generated, 5, 3, false);
    run_case(cc, ks, cp, generated, 4, 3, false);
    std::cout << "D8 PASS: " << comparisons << " residue comparisons; canonical-lift witnesses="
              << noncanonical_witnesses << "; wrong-P-offset witnesses=" << wrong_key_witnesses << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "HYBRID GRAPH FAILED: " << e.what() << '\n';return 1;
  }
}
