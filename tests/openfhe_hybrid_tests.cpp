// D8: pinned OpenFHE Hybrid graph against unreduced-CRT and scalar MAC oracles.
// Every polynomial has N=65536. Compare the resident host or CUDA transport.
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include "keyswitch/keyswitch-hybrid.h"
#include "ntt/ntt.hpp"
#include "ntt/basis_conversion.hpp"
#include "ntt/backend_contract.hpp"
#include "ntt/openfhe_hybrid_adapter.hpp"
#include "ntt/resident.hpp"
#ifndef NTT_RESIDENT_CUDA
#include "support/resident_host_driver.hpp"
#endif

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
std::vector<uint64_t> pack_evaluations(const DCRTPoly& polynomial,
                                       const ntt::proposal::PolynomialLayout& layout) {
  require(polynomial.GetFormat() == Format::EVALUATION &&
              polynomial.GetNumOfElements() == layout.towers.size(),
          "OpenFHE evaluation layout");
  std::vector<uint64_t> values;
  values.reserve(layout.towers.size() * N);
  for (size_t i = 0; i < layout.towers.size(); ++i) {
    const auto& tower = polynomial.GetElementAtIndex(i);
    const auto expected = layout.parameters->tower(layout.towers[i]);
    require(tower.GetFormat() == Format::EVALUATION &&
                tower.GetModulus().ConvertToInt() == expected.q &&
                tower.GetRootOfUnity().ConvertToInt() == expected.psi,
            "OpenFHE evaluation tower ordering");
    for (size_t coefficient = 0; coefficient < N; ++coefficient)
      values.push_back(tower[coefficient].ConvertToInt());
  }
  return values;
}
ntt::resident::Session make_resident_session(const ntt::proposal::PolynomialLayout &layout) {
#ifdef NTT_RESIDENT_CUDA
  return ntt::resident::make_cuda_session(layout);
#else
  return ntt::resident::Session(std::make_shared<resident_test::HostDriver>(layout));
#endif
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
  // The reusable candidate must preserve every residue of the independent
  // unreduced BigInteger lift, including shortened Hybrid digits.
  std::vector<uint64_t> source_moduli, target_moduli, input;
  for (size_t tower = 0; tower < source.size(); ++tower) {
    source_moduli.push_back(modulus(from, tower));
    input.insert(input.end(), source[tower].begin(), source[tower].end());
  }
  for (size_t tower = 0; tower < result.size(); ++tower)
    target_moduli.push_back(modulus(to, tower));
  const ntt::ApproximateBasisPlan candidate(std::move(source_moduli), std::move(target_moduli));
  const auto converted = candidate.convert(N, input);
  for (size_t tower = 0; tower < result.size(); ++tower)
    for (size_t index = 0; index < N; ++index)
      require(converted[tower * N + index] == result[tower][index],
              "candidate basis switch differs from independent Hybrid CRT lift");
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

// Full-prefix conversion is a separate API from Hybrid's per-digit lift.
void check_basis_conversion(const std::shared_ptr<CryptoParametersRNS>& cp) {
  const auto p = cp->GetParamsP();
  const size_t special_count = p->GetParams().size();
  const BigInteger barrett_base = BigInteger(1).LShiftEq(128);
  std::vector<DoubleNativeInt> barrett;
  for (size_t j = 0; j < special_count; ++j)
    barrett.push_back((barrett_base / BigInteger(modulus(p, j))).ConvertToInt<DoubleNativeInt>());

  for (size_t active_count = 1; active_count <= cp->GetElementParams()->GetParams().size(); ++active_count) {
    const auto active = select(cp->GetElementParams(), 0, active_count);
    std::vector<NativeInteger> inverses, preconditions;
    std::vector<std::vector<NativeInteger>> hats_mod_p(active_count);
    // Build the OpenFHE routine's parameter tables independently of lift().
    // Each hat is the product of every other active prime.
    for (size_t i = 0; i < active_count; ++i) {
      const NativeInteger qi(modulus(active, i));
      NativeInteger hat_mod_qi(1);
      for (size_t k = 0; k < active_count; ++k)
        if (k != i) hat_mod_qi = hat_mod_qi.ModMul(NativeInteger(modulus(active, k)), qi);
      inverses.push_back(hat_mod_qi.ModInverse(qi));
      preconditions.push_back(inverses.back().PrepModMulConst(qi));
      for (size_t j = 0; j < special_count; ++j) {
        const uint64_t pj = modulus(p, j);
        uint64_t hat_mod_pj = 1;
        for (size_t k = 0; k < active_count; ++k)
          if (k != i) hat_mod_pj = ntt::mul_mod(hat_mod_pj, modulus(active, k) % pj, pj);
        hats_mod_p[i].emplace_back(hat_mod_pj);
      }
    }

    for (unsigned pattern = 0; pattern < 4; ++pattern) {
      const Rows coeff = fixture(active, pattern);
      const Rows evaluation = transform(coeff, active);
      const auto ext = make(evaluation, active, Format::EVALUATION).GetExtendedCRTBasis(p);
      const Rows lifted = transform(lift(coeff, active, p), p);
      Rows expected = evaluation;
      expected.insert(expected.end(), lifted.begin(), lifted.end());
      for (const auto format : {Format::COEFFICIENT, Format::EVALUATION}) {
        auto up = make(format == Format::COEFFICIENT ? coeff : evaluation, active, format);
        up.ApproxModUp(active, p, ext, inverses, preconditions, hats_mod_p, barrett);
        equal(up, expected, ext, Format::EVALUATION, "standalone ApproxModUp");
      }

      // Arbitrary QP values exercise the P correction independently of ModUp.
      const Rows ext_eval = transform(fixture(ext, pattern), ext);
      const auto input = make(ext_eval, ext, Format::EVALUATION);
      check_down(input, ext_eval, active, cp);
      equal(input, ext_eval, ext, Format::EVALUATION, "standalone ApproxModDown preserves source");
    }
    std::cout << "BASIS CONVERSION OK: N=" << N << " active Q=" << active_count << std::endl;
  }
  require(noncanonical_witnesses > 0, "basis fixtures must detect canonical-CRT substitution");
  std::cout << "BASIS CONVERSION PASS: " << comparisons
            << " residue comparisons; canonical-lift witnesses=" << noncanonical_witnesses << std::endl;
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
void check_transactional_wrapper(const CryptoContext<DCRTPoly> &cc, const KeySwitchHYBRID &ks,
                                 const EvalKey<DCRTPoly> &key, const DCRTPoly &input,
                                 const Rows &base, bool test_failures) {
  for (const size_t count : {size_t(2), size_t(3)}) {
    auto ciphertext = std::make_shared<CiphertextImpl<DCRTPoly>>(cc, "original-key-tag");
    const auto &params = input.GetParams();
    std::vector<DCRTPoly> elements(count - 1, make(base, params, Format::COEFFICIENT));
    elements.push_back(input);
    ciphertext->SetElements(std::move(elements));
    ciphertext->SetLevel(1);
    ciphertext->SetNoiseScaleDeg(2);
    ciphertext->SetScalingFactor(1234.5);
    auto expected = ciphertext->Clone();
    ks.KeySwitchInPlace(expected, key);
    const auto original = ciphertext.get();

#ifndef NTT_RESIDENT_CUDA
    if (test_failures && count == 3) {
      const auto saved = ciphertext->GetElements();
      auto malformed = ciphertext->Clone();
      malformed->GetElements()[1].GetAllElements().front().OverrideFormat(Format::EVALUATION);
      bool opened = false;
      auto count_factory = [&](const ntt::proposal::PolynomialLayout &layout) {
        opened = true;
        return make_resident_session(layout);
      };
      bool rejected = false;
      try {
        ntt::openfhe::hybrid_key_switch_in_place(malformed, key, count_factory);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      require(rejected && !opened, "invalid leading component rejected before submission");
      auto fail_factory = [](const ntt::proposal::PolynomialLayout &layout) {
        auto driver = std::make_shared<resident_test::HostDriver>(layout);
        driver->trace->cleanup_error = true;
        return ntt::resident::Session(std::move(driver));
      };
      bool failed = false;
      try {
        ntt::openfhe::hybrid_key_switch_in_place(ciphertext, key, fail_factory);
      } catch (const std::runtime_error &error) {
        failed = std::string(error.what()).find("cleanup") != std::string::npos;
      }
      require(failed && ciphertext->GetElements() == saved && ciphertext.get() == original &&
                  ciphertext->GetLevel() == 1 && ciphertext->GetNoiseScaleDeg() == 2 &&
                  ciphertext->GetScalingFactor() == 1234.5,
              "late failure leaves ciphertext components and metadata unchanged");
    }
#else
    (void)test_failures;
#endif

    ntt::openfhe::hybrid_key_switch_in_place(ciphertext, key, make_resident_session);
    require(ciphertext.get() == original && ciphertext->GetElements().size() == 2 &&
                ciphertext->GetElements() == expected->GetElements(),
            "transactional wrapper matches pinned OpenFHE B/A add and replace semantics");
    require(ciphertext->GetLevel() == expected->GetLevel() &&
                ciphertext->GetNoiseScaleDeg() == expected->GetNoiseScaleDeg() &&
                ciphertext->GetScalingFactor() == expected->GetScalingFactor() &&
                ciphertext->GetKeyTag() == expected->GetKeyTag(),
            "transactional wrapper preserves ciphertext metadata");
    comparisons += 2 * N * input.GetNumOfElements();
  }
}
void check_decoded_slots(const CryptoContext<DCRTPoly> &cc,
                         const PrivateKey<DCRTPoly> &secret, const Ciphertext<DCRTPoly> &ciphertext,
                         const std::vector<double> &values) {
  Plaintext decrypted;
  require(cc->Decrypt(secret, ciphertext, &decrypted).isValid && decrypted,
          "decrypt the resulting CKKS ciphertext");
  decrypted->SetLength(values.size());
  const auto result = decrypted->GetRealPackedValue();
  require(result.size() == values.size(), "decoded CKKS slot count");
  for (size_t slot = 0; slot < values.size(); ++slot)
    require(std::abs(result[slot] - values[slot]) <= 1e-6, "switched CKKS slot error");
}
void check_encrypted_key_switch(const CryptoContext<DCRTPoly> &cc,
                                const KeyPair<DCRTPoly> &old_keys,
                                const KeyPair<DCRTPoly> &new_keys,
                                const EvalKey<DCRTPoly> &key) {
  const std::vector<double> values{0.25, -0.5, 1.25, -2.0};
  const auto original = cc->Encrypt(old_keys.publicKey, cc->MakeCKKSPackedPlaintext(values));
  auto ciphertext = original->Clone();
  auto expected = original->Clone();
  cc->KeySwitchInPlace(expected, key);
  ntt::openfhe::hybrid_key_switch_in_place(ciphertext, key, make_resident_session);
  require(ciphertext->GetElements() == expected->GetElements(),
          "encrypted CKKS key switch matches OpenFHE ciphertext residues");
  check_decoded_slots(cc, new_keys.secretKey, ciphertext, values);
  std::cout << "CKKS KEY SWITCH OK: exact ciphertext and decoded slots within 1e-6\n";

  cc->EvalMultKeysGen(old_keys.secretKey);
  auto product = cc->EvalMultNoRelin(original, original);
  require(product->GetElements().size() == 3, "encrypted product has three components");
  auto cube = cc->EvalMultNoRelin(original, product);
  require(cube->GetElements().size() == 4, "encrypted cube has four components");
  auto expected_product = product->Clone();
  auto expected_cube = cube->Clone();
  cc->RelinearizeInPlace(expected_product);
  cc->RelinearizeInPlace(expected_cube);

#ifndef NTT_RESIDENT_CUDA
  const auto original_components = cube->GetElements();
  size_t opened = 0;
  auto fail_second = [&](const ntt::proposal::PolynomialLayout &layout) {
    auto driver = std::make_shared<resident_test::HostDriver>(layout);
    driver->trace->cleanup_error = ++opened == 2;
    return ntt::resident::Session(driver);
  };
  bool cleanup_failed = false;
  try {
    ntt::openfhe::hybrid_relinearize_in_place(cube, fail_second);
  } catch (const std::runtime_error &error) {
    cleanup_failed = std::string(error.what()).find("cleanup") != std::string::npos;
  }
  require(cleanup_failed && opened == 2 && cube->GetElements() == original_components,
          "failure on the second component leaves the whole ciphertext unchanged");

  auto missing_key = cube->Clone();
  missing_key->SetKeyTag("unregistered-relinearization-test-key");
  opened = 0;
  bool missing_key_rejected = false;
  try {
    ntt::openfhe::hybrid_relinearize_in_place(missing_key, fail_second);
  } catch (const std::exception &) {
    missing_key_rejected = true;
  }
  require(missing_key_rejected && opened == 0 &&
              missing_key->GetElements() == original_components,
          "missing relinearization key is rejected before session creation");
#endif

  ntt::openfhe::hybrid_relinearize_in_place(product, make_resident_session);
  require(product->GetElements() == expected_product->GetElements(),
          "encrypted CKKS square relinearization matches OpenFHE ciphertext residues");
  auto squares = values;
  for (auto &value : squares) value *= value;
  check_decoded_slots(cc, old_keys.secretKey, product, squares);
  std::cout << "CKKS RELINEARIZATION OK: exact ciphertext and decoded square within 1e-6\n";

  // Exercise format normalization with a mixture of coefficient and evaluation inputs.
  cube->GetElements()[0].SetFormat(Format::COEFFICIENT);
  cube->GetElements()[3].SetFormat(Format::COEFFICIENT);
  const auto *cube_object = cube.get();
  ntt::openfhe::hybrid_relinearize_in_place(cube, make_resident_session);
  require(cube->GetElements() == expected_cube->GetElements(),
          "encrypted CKKS cube relinearization matches OpenFHE ciphertext residues");
  require(cube.get() == cube_object && cube->GetKeyTag() == expected_cube->GetKeyTag() &&
              cube->GetLevel() == expected_cube->GetLevel() &&
              cube->GetNoiseScaleDeg() == expected_cube->GetNoiseScaleDeg() &&
              cube->GetScalingFactor() == expected_cube->GetScalingFactor(),
          "relinearization preserves ciphertext identity and metadata");
  auto cubes = squares;
  for (size_t slot = 0; slot < cubes.size(); ++slot) cubes[slot] *= values[slot];
  check_decoded_slots(cc, old_keys.secretKey, cube, cubes);
  std::cout << "CKKS RELINEARIZATION OK: exact ciphertext and decoded cube within 1e-6\n";
}
void check_encrypted_rotation(const CryptoContext<DCRTPoly> &cc,
                              const KeyPair<DCRTPoly> &keys) {
  constexpr size_t slots = 8;
  const std::vector<double> values{0.25, -0.5, 1.25, -2.0, 0.75, 0.125, 3.0, -1.0};
  cc->EvalRotateKeyGen(keys.secretKey, {1, -1});
  for (const uint32_t level : {0u, 2u}) {
    const auto input = cc->Encrypt(
        keys.publicKey, cc->MakeCKKSPackedPlaintext(values, 1, level, nullptr, slots));
    require(input->GetElements()[0].GetNumOfElements() == 5 - level,
            "rotation fixture uses the intended active Q level");
    const auto original = input->Clone();
#ifndef NTT_RESIDENT_CUDA
    std::vector<std::shared_ptr<resident_test::Trace>> traces;
    auto observe_session = [&](const ntt::proposal::PolynomialLayout &layout) {
      auto driver = std::make_shared<resident_test::HostDriver>(layout);
      traces.push_back(driver->trace);
      return ntt::resident::Session(std::move(driver));
    };
#else
    const auto observe_session = make_resident_session;
#endif
    for (const int32_t shift : {1, -1}) {
      const auto expected = cc->EvalRotate(input, shift);
      const auto result = ntt::openfhe::hybrid_rotate(input, shift, observe_session);
      require(*result == *expected, "encrypted rotation matches OpenFHE residues and metadata");
      auto decoded = values;
      const size_t left = shift > 0 ? 1 : slots - 1;
      std::rotate(decoded.begin(), decoded.begin() + left, decoded.end());
      check_decoded_slots(cc, keys.secretKey, result, decoded);
      std::cout << "CKKS ROTATION OK: level=" << level << " shift=" << shift
                << " exact ciphertext and decoded slots within 1e-6\n";
    }
    require(*input == *original, "rotation preserves its input ciphertext");

    const std::array<int32_t, 3> shifts{1, 0, -1};
    const auto digits = cc->EvalFastRotationPrecompute(input);
    const auto results = ntt::openfhe::hybrid_rotate_many(input, shifts, observe_session);
    require(results.size() == shifts.size(), "hoisted rotation returns every requested shift");
    for (size_t index = 0; index < shifts.size(); ++index) {
      const auto expected = cc->EvalFastRotation(input, static_cast<uint32_t>(shifts[index]), digits);
      require(*results[index] == *expected, "hoisted rotation matches OpenFHE in request order");
      auto decoded = values;
      const size_t left = (slots + shifts[index]) % slots;
      std::rotate(decoded.begin(), decoded.begin() + left, decoded.end());
      check_decoded_slots(cc, keys.secretKey, results[index], decoded);
    }
    require(*input == *original && results[1].get() != input.get(),
            "hoisted rotation preserves the input and clones the zero shift");
    std::cout << "CKKS HOISTED ROTATION OK: level=" << level << " shifts=1,0,-1\n";
#ifndef NTT_RESIDENT_CUDA
    require(traces.size() == 3, "two single rotations and one shared hoisted session");
    const auto separate_operations = traces[0]->executed + traces[1]->executed;
    require(traces[2]->executed < separate_operations,
            "hoisted rotation reuses work across nonzero shifts");
    std::cout << "HOISTED HOST OPERATIONS: level=" << level << " separate=" << separate_operations
              << " hoisted=" << traces[2]->executed << '\n';
#endif

    if (level != 0)
      continue;
    bool opened = false;
    auto count_factory = [&](const ntt::proposal::PolynomialLayout &layout) {
      opened = true;
      return make_resident_session(layout);
    };
    bool missing_key_rejected = false;
    try {
      (void)ntt::openfhe::hybrid_rotate(input, 2, count_factory);
    } catch (const std::invalid_argument &) {
      missing_key_rejected = true;
    }
    require(missing_key_rejected && !opened, "rotation rejects missing keys before session creation");
    const std::array<int32_t, 2> missing_shift{1, 2};
    missing_key_rejected = false;
    try {
      (void)ntt::openfhe::hybrid_rotate_many(input, missing_shift, count_factory);
    } catch (const std::invalid_argument &) {
      missing_key_rejected = true;
    }
    require(missing_key_rejected && !opened, "hoisted rotation preflights keys across the batch");
    const std::array<int32_t, 2> identity_shifts{0, 0};
    auto unregistered = input->Clone();
    unregistered->SetKeyTag("unregistered-identity-rotation-test-key");
    const auto identities = ntt::openfhe::hybrid_rotate_many(unregistered, identity_shifts, count_factory);
    require(!opened && identities.size() == 2 && *identities[0] == *unregistered &&
                *identities[1] == *unregistered && identities[0] != identities[1],
            "identity-only batch clones the input without a key or session");
#ifndef NTT_RESIDENT_CUDA
    auto fail_factory = [](const ntt::proposal::PolynomialLayout &layout) {
      auto driver = std::make_shared<resident_test::HostDriver>(layout);
      driver->trace->cleanup_error = true;
      return ntt::resident::Session(std::move(driver));
    };
    bool cleanup_failed = false;
    try {
      (void)ntt::openfhe::hybrid_rotate(input, 1, fail_factory);
    } catch (const std::runtime_error &error) {
      cleanup_failed = std::string(error.what()).find("cleanup") != std::string::npos;
    }
    require(cleanup_failed && *input == *original, "failed rotation leaves input unchanged");
    cleanup_failed = false;
    std::vector<Ciphertext<DCRTPoly>> failed_batch;
    try {
      failed_batch = ntt::openfhe::hybrid_rotate_many(input, shifts, fail_factory);
    } catch (const std::runtime_error &error) {
      cleanup_failed = std::string(error.what()).find("cleanup") != std::string::npos;
    }
    require(cleanup_failed && failed_batch.empty() && *input == *original,
            "failed hoisted rotation leaves the input unchanged and publishes no results");
#endif
  }
}

unsigned benchmark_count(const char *argument, unsigned minimum, unsigned maximum) {
  const std::string_view text(argument);
  unsigned count = 0;
  const auto end = text.data() + text.size();
  const auto parsed = std::from_chars(text.data(), end, count);
  require(parsed.ec == std::errc{} && parsed.ptr == end && count >= minimum && count <= maximum,
          "rotation benchmark requires 0..20 warmup pairs and 2..100 measured pairs");
  return count;
}

constexpr std::array<uint32_t, 2> rotation_benchmark_levels{0, 2};
constexpr const char *rotation_fixture_header = "NTT CKKS rotation benchmark v1";
const std::vector<int32_t> rotation_benchmark_shifts{1, -1, 2, -2};

void prepare_rotation_fixture(const CryptoContext<DCRTPoly> &cc, const char *path) {
  const std::vector<double> slots{0.25, -0.5, 1.25, -2.0, 0.75, 0.125, 3.0, -1.0};
  const auto keys = cc->KeyGen();
  cc->EvalRotateKeyGen(keys.secretKey, rotation_benchmark_shifts);
  require(!std::filesystem::exists(path), "rotation fixture path already exists");
  std::ofstream fixture(path, std::ios::binary);
  require(bool(fixture), "cannot create rotation fixture");
  fixture << rotation_fixture_header << '\n';
  require(cc->SerializeEvalAutomorphismKey(fixture, SerType::BINARY, keys.secretKey->GetKeyTag()),
          "cannot serialize rotation evaluation keys");
  for (const auto level : rotation_benchmark_levels) {
    const auto input = cc->Encrypt(
        keys.publicKey, cc->MakeCKKSPackedPlaintext(slots, 1, level, nullptr, slots.size()));
    Serial::Serialize(input, fixture, SerType::BINARY);
    for (const auto shift : rotation_benchmark_shifts)
      Serial::Serialize(cc->EvalRotate(input, shift), fixture, SerType::BINARY);
  }
  fixture.close();
  require(bool(fixture), "cannot write rotation fixture");
}

void benchmark_rotations(const CryptoContext<DCRTPoly> &cc, const char *path,
                         unsigned warmups, unsigned pairs) {
  using Clock = std::chrono::steady_clock;
  const auto &shifts = rotation_benchmark_shifts;
  std::ifstream fixture(path, std::ios::binary);
  std::string header;
  require(bool(std::getline(fixture, header)) && header == rotation_fixture_header,
          "invalid rotation fixture header");
  require(cc->DeserializeEvalAutomorphismKey(fixture, SerType::BINARY),
          "cannot load rotation evaluation keys");
#ifdef NTT_RESIDENT_CUDA
  constexpr auto backend = "cuda";
#else
  constexpr auto backend = "host_simulation";
#endif
  for (const auto level : rotation_benchmark_levels) {
    Ciphertext<DCRTPoly> input;
    Serial::Deserialize(input, fixture, SerType::BINARY);
    require(input && input->GetCryptoContext() == cc && input->GetLevel() == level &&
                input->GetElements().size() == 2, "rotation benchmark ciphertext context or level");
    const auto active_q = input->GetElements()[0].GetNumOfElements();
    require(active_q == 5 - level, "rotation benchmark active Q level");
    std::vector<Ciphertext<DCRTPoly>> expected(shifts.size());
    for (auto &reference : expected) {
      Serial::Deserialize(reference, fixture, SerType::BINARY);
      require(bool(reference), "missing rotation reference ciphertext");
    }
    auto measure = [&](bool hoisted) {
      std::vector<Ciphertext<DCRTPoly>> outputs;
      const auto start = Clock::now();
      if (hoisted) {
        outputs = ntt::openfhe::hybrid_rotate_many(input, shifts, make_resident_session);
      } else {
        outputs.reserve(shifts.size());
        for (const auto shift : shifts)
          outputs.push_back(ntt::openfhe::hybrid_rotate(input, shift, make_resident_session));
      }
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start);
      require(outputs.size() == expected.size(), "paired rotation output count");
      for (size_t shift = 0; shift < shifts.size(); ++shift)
        require(*outputs[shift] == *expected[shift], "paired rotation differs from OpenFHE");
      return elapsed.count();
    };
    for (unsigned round = 0; round < warmups + pairs; ++round) {
      const bool hoisted_first = round % 2 != 0;
      const auto first = measure(hoisted_first);
      const auto second = measure(!hoisted_first);
      if (round < warmups)
        continue;
      std::cout << "{\"schema\":1,\"backend\":\"" << backend
                << "\",\"timing\":\"public_call_wall\",\"ring_dimension\":" << N
                << ",\"active_q\":" << active_q << ",\"shifts\":[1,-1,2,-2],\"pair\":"
                << round - warmups + 1 << ",\"order\":\""
                << (hoisted_first ? "hoisted_first" : "individual_first")
                << "\",\"individual_ns\":" << (hoisted_first ? second : first)
                << ",\"hoisted_ns\":" << (hoisted_first ? first : second) << "}" << std::endl;
    }
  }
  require(fixture.peek() == std::char_traits<char>::eof() && !fixture.bad(),
          "unexpected bytes after rotation fixture");
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
              size_t L, unsigned pattern, bool wrappers, bool adapter_at_every_level = false) {
  const auto active = select(cp->GetElementParams(), 0, L);
  const Rows coeff = fixture(active, pattern), evaluation = transform(coeff, active);
  const auto input = make(evaluation, active, Format::EVALUATION);
  const auto digits = ks.EvalKeySwitchPrecomputeCore(input, cp);
  const auto expected_digits = check_digits(input, coeff, cp, digits);
  const auto ext = ks.EvalFastKeySwitchCoreExt(digits, key, active);
  const auto fast = ks.EvalFastKeySwitchCore(digits, key, active);
  // D9 bridge: derive metadata from real OpenFHE parameters, never an object
  // address or a presumed contiguous NativePoly allocation.
  {
    namespace contract = ntt::proposal;
    std::vector<contract::Tower> q, p;
    const auto full_q = cp->GetElementParams(), special = cp->GetParamsP();
    for(size_t i=0;i<full_q->GetParams().size();++i) q.push_back({modulus(full_q,i),root(full_q,i)});
    for(size_t i=0;i<special->GetParams().size();++i) p.push_back({modulus(special,i),root(special,i)});
    const auto snapshot = contract::Parameters::create(N,q,p,cp->GetNumPerPartQ());
    const contract::PolynomialLayout layout{snapshot,L,contract::qp_basis(*snapshot,L),
      contract::Domain::Evaluation,contract::Order::BitReversed};
    contract::validate(layout);
    const auto offsets=contract::key_indices(*snapshot,L);
    for(size_t i=0;i<layout.towers.size();++i) {
      const auto expected=snapshot->tower(layout.towers[i]);
      require(expected.q==modulus(ext[0].GetParams(),i) && expected.psi==root(ext[0].GetParams(),i),
              "D9 active QlP metadata matches actual Hybrid output");
      require(expected.q==modulus(key->GetAVector()[0].GetParams(),offsets[i]) &&
              expected.psi==root(key->GetAVector()[0].GetParams(),offsets[i]),
              "D9 key view addresses the full-context QP snapshot");
    }

    const auto coefficient_layout = contract::transform_layout(layout, true);
    auto resident = make_resident_session(coefficient_layout);
    const contract::PolynomialLayout input_layout{snapshot, L, contract::q_prefix(*snapshot, L),
      contract::Domain::Evaluation, contract::Order::BitReversed};
    const auto input_values = pack_evaluations(input, input_layout);
    const auto resident_input = resident.upload(input_layout, input_values);
    const auto resident_digits = resident.hybrid_digits(resident_input);
    require(resident_digits.size() == expected_digits.size(), "resident Hybrid digit count");
    require(resident.stats().uploads == 1 && resident.stats().downloads == 0,
            "resident Hybrid precompute has no intermediate host transfers");
    for (size_t digit = 0; digit < resident_digits.size(); ++digit) {
      contract::require_layout(resident_digits[digit].layout(), layout);
      const auto download = resident.download(resident_digits[digit]);
      const auto values = download.wait_values();
      for (size_t tower = 0; tower < layout.towers.size(); ++tower) {
        const auto& expected = expected_digits[digit][tower];
        require(std::equal(expected.begin(), expected.end(), values.begin() + tower * N),
                "resident Hybrid digit residues differ from OpenFHE");
        comparisons += N;
      }
    }
    const auto full_count = snapshot->q().size();
    const contract::PolynomialLayout full_key_layout{snapshot, full_count,
      contract::qp_basis(*snapshot, full_count), contract::Domain::Evaluation,
      contract::Order::BitReversed};
    auto upload_key = [&] {
      std::vector<std::array<std::vector<uint64_t>, 2>> storage;
      for (size_t digit = 0; digit < key->GetBVector().size(); ++digit)
        storage.push_back({pack_evaluations(key->GetBVector()[digit], full_key_layout),
                           pack_evaluations(key->GetAVector().at(digit), full_key_layout)});
      std::vector<ntt::resident::HybridKeyDigit> rows;
      for (const auto& row : storage) rows.push_back({row[0], row[1]});
      return resident.upload_hybrid_key(full_key_layout, rows);
    };
    const auto resident_key = upload_key();
    const auto transfers_before = resident.stats();
    require(transfers_before.uploads == 1 + 2 * resident_digits.size(),
            "only active key digits are uploaded");
    const auto mac = resident.hybrid_accumulate(resident_digits, resident_key);
    const auto reduced_b = resident.mod_down(mac.b);
    const auto reduced_a = resident.mod_down(mac.a);
    require(resident.stats().downloads == transfers_before.downloads &&
                resident.stats().uploads == transfers_before.uploads,
            "Hybrid MAC and ModDown have no intermediate polynomial transfers");
    auto compare_output = [&](const ntt::resident::Polynomial& result,
                              const DCRTPoly& expected,
                              const contract::PolynomialLayout& result_layout) {
      contract::require_layout(result.layout(), result_layout);
      const auto download = resident.download(result);
      const auto reference = pack_evaluations(expected, result_layout);
      require(std::ranges::equal(download.wait_values(), reference),
              "resident Hybrid contribution differs from OpenFHE");
      comparisons += reference.size();
    };
    compare_output(mac.b, ext[0], layout);
    compare_output(mac.a, ext[1], layout);
    compare_output(reduced_b, fast[0], input_layout);
    compare_output(reduced_a, fast[1], input_layout);
    const auto original = resident.download(resident_input);
    require(std::ranges::equal(original.wait_values(), input_values),
            "resident Hybrid source remains unchanged");
    resident.close();
  }
  const auto expected_ext = check_mac(expected_digits, key, active, ext);
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
  if ((L == 4 || adapter_at_every_level) && pattern == 3) {
    const auto result = ntt::openfhe::hybrid_key_switch_core(input, key, make_resident_session);
    equal(result.b, down[0], active, Format::EVALUATION, "adapter B contribution");
    equal(result.a, down[1], active, Format::EVALUATION, "adapter A contribution");
    check_transactional_wrapper(cc, ks, key, input, coeff, wrappers);
  }
#ifndef NTT_RESIDENT_CUDA
  if (wrappers && L == 4 && pattern == 3) {
    auto malformed = input;
    malformed.GetAllElements().front().OverrideFormat(Format::COEFFICIENT);
    bool factory_called = false;
    auto count_factory = [&](const ntt::proposal::PolynomialLayout &layout) {
      factory_called = true;
      return make_resident_session(layout);
    };
    bool invalid_rejected = false;
    try {
      (void)ntt::openfhe::hybrid_key_switch_core(malformed, key, count_factory);
    } catch (const std::invalid_argument &) {
      invalid_rejected = true;
    }
    require(invalid_rejected && !factory_called, "adapter rejects mixed formats before session creation");

    auto bad_key = std::make_shared<EvalKeyRelinImpl<DCRTPoly>>(cc);
    bad_key->SetAVector(key->GetAVector());
    auto short_b = key->GetBVector();
    short_b.pop_back();
    bad_key->SetBVector(std::move(short_b));
    invalid_rejected = false;
    try {
      (void)ntt::openfhe::hybrid_key_switch_core(input, bad_key, count_factory);
    } catch (const std::invalid_argument &) {
      invalid_rejected = true;
    }
    require(invalid_rejected && !factory_called,
            "adapter rejects incomplete key channels before session creation");

    std::shared_ptr<resident_test::HostDriver> driver;
    auto failing_factory = [&](const ntt::proposal::PolynomialLayout &layout) {
      driver = std::make_shared<resident_test::HostDriver>(layout);
      driver->trace->cleanup_error = true;
      return ntt::resident::Session(driver);
    };
    bool close_failed = false;
    try {
      (void)ntt::openfhe::hybrid_key_switch_core(input, key, failing_factory);
    } catch (const std::runtime_error &error) {
      close_failed = std::string(error.what()).find("cleanup") != std::string::npos;
    }
    require(close_failed && driver && driver->trace->executed > 0 &&
                driver->trace->live_buffers == 0 && driver->trace->live_events == 0,
            "adapter propagates late session failure without returning output or retaining buffers");
  }
#endif
  equal(input, evaluation, active, Format::EVALUATION, "source input remains unchanged");
  std::cout << "HYBRID GRAPH OK: N=" << N << " L=" << L << " pattern=" << pattern
            << " digits=" << digits->size() << " key=" << (wrappers ? "diagnostic+wrapper" : "generated") << std::endl;
}

// Keep the original pinned-parameter fixture and serialized benchmark stable.
// This separate gate varies the values that determine digit and key addressing.
void check_parameter_matrix() {
#ifdef NTT_RESIDENT_CUDA
  std::cout << "HYBRID PARAMETER MATRIX BACKEND: cuda" << std::endl;
#else
  std::cout << "HYBRID PARAMETER MATRIX BACKEND: host_simulation" << std::endl;
#endif
  struct Configuration {
    uint32_t scaling_bits, first_bits, large_digits, expected_alpha;
  };
  constexpr std::array<Configuration, 3> configurations{{
      // OpenFHE 1.6.1 requires scaling bits < 60 for NativeInteger 64.
      {30, 31, 3, 2}, {50, 60, 5, 1}, {59, 60, 2, 3}}};
  KeySwitchHYBRID ks;
  size_t cases = 0;
  for (const auto &configuration : configurations) {
    const auto canonical_before = noncanonical_witnesses, offset_before = wrong_key_witnesses;
    CCParams<CryptoContextCKKSRNS> config;
    config.SetRingDim(N);
    config.SetMultiplicativeDepth(4);
    config.SetScalingModSize(configuration.scaling_bits);
    config.SetFirstModSize(configuration.first_bits);
    config.SetScalingTechnique(FIXEDMANUAL);
    config.SetKeySwitchTechnique(HYBRID);
    config.SetNumLargeDigits(configuration.large_digits);
    const auto cc = GenCryptoContext(config);
    cc->Enable(PKE);cc->Enable(KEYSWITCH);cc->Enable(LEVELEDSHE);
    const auto cp = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());
    require(cc->GetRingDimension() == N && cp->GetElementParams()->GetParams().size() == 5 &&
                cp->GetNumPerPartQ() == configuration.expected_alpha && cp->GetNoiseScale() == 1,
            "parameter matrix configuration");
    std::cout << "PARAMETER MATRIX CONTEXT: N=" << N
              << " scaling_bits=" << configuration.scaling_bits
              << " first_bits=" << configuration.first_bits
              << " M=5 alpha=" << cp->GetNumPerPartQ()
              << " K=" << cp->GetParamsP()->GetParams().size() << std::endl;
    for (size_t tower = 0; tower < cp->GetParamsQP()->GetParams().size(); ++tower)
      std::cout << "PARAM " << tower << " q=" << modulus(cp->GetParamsQP(), tower)
                << " psi=" << root(cp->GetParamsQP(), tower) << std::endl;

    const auto diagnostic = diagnostic_key(cc, cp);
    for (size_t active = 1; active <= 5; ++active) {
      run_case(cc, ks, cp, diagnostic, active, 3, true, true);
      ++cases;
    }
    // Actual evaluation-key arrays must work at both ends of the level range.
    // Compare the same generated arrays in OpenFHE and the resident transport.
    const auto old_keys = cc->KeyGen(), new_keys = cc->KeyGen();
    const auto generated = cc->KeySwitchGen(old_keys.secretKey, new_keys.secretKey);
    for (const size_t active : {size_t(1), size_t(5)}) {
      run_case(cc, ks, cp, generated, active, 3, false, true);
      ++cases;
    }
    require(noncanonical_witnesses > canonical_before && wrong_key_witnesses > offset_before,
            "each context must detect canonical CRT and wrong reduced-level key offsets");
  }
  require(cases == 21 && noncanonical_witnesses > 0 && wrong_key_witnesses > 0,
          "parameter matrix must cover every case and detect incorrect alternatives");
  std::cout << "HYBRID PARAMETER MATRIX PASS: " << cases << " cases; " << comparisons
            << " residue comparisons; canonical-lift witnesses=" << noncanonical_witnesses
            << "; wrong-P-offset witnesses=" << wrong_key_witnesses << std::endl;
}
} // namespace

int main(int argc, char** argv) {
  try {
    const bool basis_only = argc == 2 && std::string(argv[1]) == "--basis-only";
    const bool parameter_matrix = argc == 2 && std::string_view(argv[1]) == "--parameter-matrix";
    const bool rotation_prep = argc == 3 && std::string_view(argv[1]) == "--prepare-rotation";
    const bool rotation_benchmark = argc == 5 && std::string_view(argv[1]) == "--rotation-benchmark";
    require(argc == 1 || basis_only || parameter_matrix || rotation_prep || rotation_benchmark,
            "usage: ntt_openfhe_hybrid [--basis-only | --parameter-matrix | --prepare-rotation FIXTURE | "
            "--rotation-benchmark FIXTURE WARMUPS PAIRS]");
    if (parameter_matrix) {
      check_parameter_matrix();
      return 0;
    }
    const auto warmups = rotation_benchmark ? benchmark_count(argv[3], 0, 20) : 0;
    const auto pairs = rotation_benchmark ? benchmark_count(argv[4], 2, 100) : 0;
    CCParams<CryptoContextCKKSRNS> config;
    config.SetRingDim(N);config.SetMultiplicativeDepth(4);config.SetScalingModSize(50);
    config.SetFirstModSize(60);config.SetScalingTechnique(FIXEDMANUAL);
    config.SetKeySwitchTechnique(HYBRID);config.SetNumLargeDigits(2);
    config.SetMaxRelinSkDeg(3);
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
    if (rotation_prep || rotation_benchmark) {
      if (rotation_prep)
        prepare_rotation_fixture(cc, argv[2]);
      else
        benchmark_rotations(cc, argv[2], warmups, pairs);
      return 0;
    }
    std::cout << "CONTEXT N=" << N << " M=5 alpha=3 beta=2 K=" << cp->GetParamsP()->GetParams().size() << std::endl;
    for (size_t l = 0; l < cp->GetParamsQP()->GetParams().size(); ++l)
      std::cout << "PARAM " << l << " q=" << modulus(cp->GetParamsQP(), l)
                << " psi=" << root(cp->GetParamsQP(), l) << std::endl;
    if (basis_only) {
      check_basis_conversion(cp);
      return 0;
    }
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
    check_encrypted_key_switch(cc, old_keys, new_keys, generated);
    check_encrypted_rotation(cc, old_keys);
    std::cout << "D8 PASS: " << comparisons << " residue comparisons; canonical-lift witnesses="
              << noncanonical_witnesses << "; wrong-P-offset witnesses=" << wrong_key_witnesses << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "HYBRID GRAPH FAILED: " << e.what() << '\n';return 1;
  }
}
