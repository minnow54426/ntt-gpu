#pragma once
// Opt-in bridge for the pinned CKKS Hybrid core. The caller supplies the
// resident-session factory and chooses whether to update a ciphertext.
#include "ntt/resident.hpp"
#include "openfhe.h"
#include <array>

namespace ntt::openfhe {
namespace detail {
using lbcrypto::DCRTPoly;

inline std::vector<proposal::Tower> snapshot_towers(const DCRTPoly::Params &params,
                                                    size_t dimension) {
  proposal::require(params.GetCyclotomicOrder() == 2 * dimension,
                    "OpenFHE parameter ring dimension mismatch");
  std::vector<proposal::Tower> towers;
  towers.reserve(params.GetParams().size());
  for (const auto &tower : params.GetParams()) {
    proposal::require(tower && tower->GetCyclotomicOrder() == 2 * dimension,
                      "OpenFHE tower ring dimension mismatch");
    towers.push_back({tower->GetModulus().ConvertToInt(), tower->GetRootOfUnity().ConvertToInt()});
  }
  return towers;
}

template <class Visit>
void inspect(const DCRTPoly &polynomial, const proposal::PolynomialLayout &layout,
             Format format, Visit visit) {
  const size_t dimension = layout.parameters->n();
  proposal::require((format == Format::EVALUATION || format == Format::COEFFICIENT) &&
                        polynomial.GetFormat() == format &&
                        polynomial.GetParams() &&
                        polynomial.GetAllElements().size() == layout.towers.size(),
                    "OpenFHE Hybrid input requires a complete polynomial in the expected format");
  const auto parameter_towers = snapshot_towers(*polynomial.GetParams(), dimension);
  proposal::require(parameter_towers.size() == layout.towers.size(),
                    "OpenFHE polynomial parameter count mismatch");
  for (size_t index = 0; index < layout.towers.size(); ++index) {
    const auto expected = layout.parameters->tower(layout.towers[index]);
    const auto &tower = polynomial.GetElementAtIndex(index);
    proposal::require(parameter_towers[index] == expected && tower.GetParams() &&
                          tower.GetParams()->GetCyclotomicOrder() == 2 * dimension &&
                          tower.GetFormat() == format &&
                          tower.GetModulus().ConvertToInt() == expected.q &&
                          tower.GetRootOfUnity().ConvertToInt() == expected.psi && !tower.IsEmpty(),
                      "OpenFHE Hybrid child tower metadata mismatch");
    const auto &values = tower.GetValues();
    proposal::require(values.GetLength() == dimension &&
                          values.GetModulus().ConvertToInt() == expected.q,
                      "OpenFHE Hybrid tower value extent or modulus mismatch");
    for (size_t coefficient = 0; coefficient < dimension; ++coefficient) {
      const auto value = values[coefficient].ConvertToInt();
      proposal::require(value < expected.q, "noncanonical OpenFHE Hybrid residue");
      visit(value);
    }
  }
}
inline std::vector<uint64_t> pack(const DCRTPoly &polynomial,
                                  const proposal::PolynomialLayout &layout) {
  std::vector<uint64_t> packed;
  packed.reserve(proposal::bytes(layout) / sizeof(uint64_t));
  inspect(polynomial, layout, Format::EVALUATION, [&](uint64_t value) { packed.push_back(value); });
  return packed;
}

inline DCRTPoly restore(const std::shared_ptr<DCRTPoly::Params> &params,
                        const proposal::PolynomialLayout &layout,
                        std::span<const uint64_t> packed) {
  using namespace lbcrypto;
  proposal::validate_host_values(layout, packed);
  const size_t dimension = layout.parameters->n();
  DCRTPoly result(params, Format::EVALUATION, false);
  for (size_t index = 0; index < layout.towers.size(); ++index) {
    const auto &tower_params = params->GetParams()[index];
    NativeVector values(dimension, tower_params->GetModulus());
    for (size_t coefficient = 0; coefficient < dimension; ++coefficient)
      values[coefficient] = NativeInteger(packed[index * dimension + coefficient]);
    NativePoly tower(tower_params, Format::EVALUATION);
    tower.SetValues(std::move(values), Format::EVALUATION);
    result.SetElementAtIndex(index, std::move(tower));
  }
  return result;
}
inline proposal::PolynomialLayout source_layout(const DCRTPoly &input,
                                                const lbcrypto::CryptoContext<DCRTPoly> &context) {
  using namespace lbcrypto;
  proposal::require(bool(context), "missing OpenFHE Hybrid context");
  const auto crypto = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(context->GetCryptoParameters());
  proposal::require(crypto && crypto->GetKeySwitchTechnique() == HYBRID &&
                        crypto->GetNoiseScale() == 1 && crypto->GetElementParams() &&
                        crypto->GetParamsP() && input.GetParams(),
                    "adapter requires CKKS Hybrid parameters and input tower metadata");
  const size_t dimension = crypto->GetElementParams()->GetRingDimension();
  auto q = detail::snapshot_towers(*crypto->GetElementParams(), dimension);
  auto p = detail::snapshot_towers(*crypto->GetParamsP(), dimension);
  const size_t active_q = input.GetParams()->GetParams().size();
  proposal::require(dimension <= 65536 && active_q > 0 && active_q <= q.size() &&
                        active_q <= 40 && !p.empty() && p.size() <= 40,
                    "unsupported resident Hybrid dimension or basis size");
  const auto parameters = proposal::Parameters::create(
      dimension, std::move(q), std::move(p), crypto->GetNumPerPartQ());
  return {parameters, active_q, proposal::q_prefix(*parameters, active_q),
          proposal::Domain::Evaluation, proposal::Order::BitReversed};
}
} // namespace detail

struct HybridContributions {
  lbcrypto::DCRTPoly b, a;
};

namespace detail {
inline proposal::PolynomialLayout resident_base(const proposal::PolynomialLayout &source) {
  const auto &parameters = source.parameters;
  return {parameters, source.active_q, proposal::qp_basis(*parameters, source.active_q),
          proposal::Domain::Coefficient, proposal::Order::Natural};
}

class PackedHybridKey {
  proposal::PolynomialLayout layout_;
  std::vector<std::array<std::vector<uint64_t>, 2>> channels_;

public:
  PackedHybridKey(const lbcrypto::EvalKey<DCRTPoly> &key, const proposal::PolynomialLayout &source)
      : layout_{source.parameters, source.parameters->q().size(),
                proposal::qp_basis(*source.parameters, source.parameters->q().size()),
                proposal::Domain::Evaluation, proposal::Order::BitReversed} {
    const auto &key_b = key->GetBVector();
    const auto &key_a = key->GetAVector();
    const size_t full_digits = 1 + (layout_.active_q - 1) / layout_.parameters->alpha();
    proposal::require(key_b.size() == full_digits && key_a.size() == full_digits,
                      "wrong OpenFHE Hybrid key channel extent");
    channels_.reserve(full_digits);
    for (size_t digit = 0; digit < full_digits; ++digit)
      channels_.push_back({pack(key_b[digit], layout_), pack(key_a[digit], layout_)});
  }

  resident::HybridKey upload(resident::Session &session) const {
    std::vector<resident::HybridKeyDigit> rows;
    rows.reserve(channels_.size());
    for (const auto &channel : channels_)
      rows.push_back({channel[0], channel[1]});
    return session.upload_hybrid_key(layout_, rows);
  }
};

inline HybridContributions reduce_and_download(resident::Session &session,
                                               const resident::HybridContributions &extended,
                                               const proposal::PolynomialLayout &source,
                                               const std::shared_ptr<DCRTPoly::Params> &params) {
  const auto reduced_b = session.mod_down(extended.b);
  const auto reduced_a = session.mod_down(extended.a);
  proposal::require_layout(reduced_b.layout(), source);
  proposal::require_layout(reduced_a.layout(), source);
  const auto download_b = session.download(reduced_b);
  const auto download_a = session.download(reduced_a);
  // Waiting for the last download retires the earlier operation pins as well.
  const auto values_a = download_a.wait_values();
  return {restore(params, source, download_b.wait_values()), restore(params, source, values_a)};
}

// Synchronous transaction. Both candidates are private until download,
// validation and session close succeed. The caller's input and key are read-only.
template <class MakeSession>
HybridContributions run_hybrid_core(const DCRTPoly &input, const lbcrypto::EvalKey<DCRTPoly> &key,
                                     const proposal::PolynomialLayout &source, MakeSession &make_session,
                                     std::span<const uint64_t> first_component = {},
                                     uint32_t automorphism = 0) {
  const auto evaluations = pack(input, source);
  const PackedHybridKey packed_key(key, source);

  auto session = make_session(resident_base(source));
  const auto resident_key = packed_key.upload(session);
  const auto digits = session.hybrid_digits(session.upload(source, evaluations));
  auto extended = session.hybrid_accumulate(digits, resident_key);
  if (!first_component.empty())
    extended = session.hybrid_rotate_extended(extended, session.upload(source, first_component),
                                              automorphism);
  auto result = reduce_and_download(session, extended, source, input.GetParams());
  session.close();
  return result;
}

inline uint32_t rotation_index(const lbcrypto::CryptoContext<DCRTPoly> &context,
                               int32_t slot_shift, int32_t dimension) {
  // Generator 5 has order N/2. Preserve OpenFHE's reserved conjugation index.
  if (slot_shift != 2 * dimension - 1)
    slot_shift %= dimension / 2;
  return context->FindAutomorphismIndex(static_cast<uint32_t>(slot_shift));
}

using RotationKeys = std::map<uint32_t, lbcrypto::EvalKey<DCRTPoly>>;
inline lbcrypto::EvalKey<DCRTPoly> rotation_key(const RotationKeys &keys, uint32_t automorphism,
                                               const lbcrypto::CryptoContext<DCRTPoly> &context) {
  const auto key = keys.find(automorphism);
  proposal::require(key != keys.end() && key->second && key->second->GetCryptoContext() == context,
                    "missing or foreign OpenFHE rotation key");
  return key->second;
}

inline lbcrypto::Ciphertext<DCRTPoly> rotation_result(
    lbcrypto::ConstCiphertext<DCRTPoly> &input, HybridContributions rotated) {
  std::vector<DCRTPoly> channels;
  channels.reserve(2);
  channels.push_back(std::move(rotated.b));
  channels.push_back(std::move(rotated.a));
  auto result = input->CloneEmpty();
  result->SetElements(std::move(channels));
  return result;
}
} // namespace detail

template <class MakeSession>
HybridContributions hybrid_key_switch_core(const lbcrypto::DCRTPoly &input,
                                            const lbcrypto::EvalKey<lbcrypto::DCRTPoly> &key,
                                            MakeSession make_session) {
  proposal::require(bool(key), "missing OpenFHE Hybrid evaluation key");
  return detail::run_hybrid_core(input, key, detail::source_layout(input, key->GetCryptoContext()),
                                 make_session);
}

template <class MakeSession>
void hybrid_key_switch_in_place(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ciphertext,
                                 const lbcrypto::EvalKey<lbcrypto::DCRTPoly> &key,
                                 MakeSession make_session) {
  proposal::require(ciphertext && ciphertext->GetCryptoContext() && key &&
                        ciphertext->GetCryptoContext() == key->GetCryptoContext(),
                    "OpenFHE ciphertext and evaluation key must share a context");
  const auto &elements = ciphertext->GetElements();
  proposal::require(elements.size() == 2 || elements.size() == 3,
                    "Hybrid wrapper requires two or three ciphertext components");
  const auto source = detail::source_layout(elements.back(), ciphertext->GetCryptoContext());
  const size_t leading = elements.size() - 1;
  for (size_t index = 0; index < leading; ++index)
    detail::inspect(elements[index], source, elements[index].GetFormat(), [](uint64_t) {});

  auto contributions = detail::run_hybrid_core(elements.back(), key, source, make_session);
  // Keep the format conversions and additions private until the noexcept commit.
  std::vector<lbcrypto::DCRTPoly> candidate(elements.begin(), elements.begin() + leading);
  candidate[0].SetFormat(Format::EVALUATION);
  candidate[0] += contributions.b;
  if (leading == 2) {
    candidate[1].SetFormat(Format::EVALUATION);
    candidate[1] += contributions.a;
  } else {
    candidate.push_back(std::move(contributions.a));
  }
  ciphertext->SetElements(std::move(candidate));
}

// Match the pinned RelinearizeInPlace key ordering. Each higher component
// contributes to private outputs before the single ciphertext commit.
template <class MakeSession>
void hybrid_relinearize_in_place(lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ciphertext,
                                 MakeSession make_session) {
  proposal::require(ciphertext && ciphertext->GetCryptoContext(), "missing OpenFHE ciphertext context");
  const auto context = ciphertext->GetCryptoContext();
  const auto &elements = ciphertext->GetElements();
  proposal::require(elements.size() >= 2, "relinearization requires at least two components");
  const auto &keys = context->GetEvalMultKeyVector(ciphertext->GetKeyTag());
  const size_t extra = elements.size() - 2;
  proposal::require(keys.size() >= extra, "insufficient OpenFHE relinearization keys");
  for (size_t index = 0; index < extra; ++index)
    proposal::require(keys[index] && keys[index]->GetCryptoContext() == context,
                      "OpenFHE relinearization key context mismatch");

  const auto source = detail::source_layout(elements.back(), context);
  for (const auto &component : elements)
    detail::inspect(component, source, component.GetFormat(), [](uint64_t) {});

  std::vector<lbcrypto::DCRTPoly> candidate(elements.begin(), elements.begin() + 2);
  candidate[0].SetFormat(Format::EVALUATION);
  candidate[1].SetFormat(Format::EVALUATION);
  for (size_t index = 0; index < extra; ++index) {
    const auto &component = elements[index + 2];
    auto contributions = [&] {
      if (component.GetFormat() == Format::EVALUATION)
        return detail::run_hybrid_core(component, keys[index], source, make_session);
      auto evaluation = component;
      evaluation.SetFormat(Format::EVALUATION);
      return detail::run_hybrid_core(evaluation, keys[index], source, make_session);
    }();
    candidate[0] += contributions.b;
    candidate[1] += contributions.a;
  }
  ciphertext->SetElements(std::move(candidate));
}

// OpenFHE applies the automorphism to the extended QP result before ModDown.
template <class MakeSession>
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> hybrid_rotate(
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly> &ciphertext, int32_t slot_shift,
    MakeSession make_session) {
  proposal::require(ciphertext && ciphertext->GetCryptoContext(), "missing OpenFHE ciphertext context");
  const auto context = ciphertext->GetCryptoContext();
  const auto &elements = ciphertext->GetElements();
  proposal::require(elements.size() == 2, "Hybrid rotation requires a relinearized ciphertext");
  const auto source = detail::source_layout(elements[1], context);
  const auto automorphism = detail::rotation_index(
      context, slot_shift, static_cast<int32_t>(source.parameters->n()));
  const auto keys = context->GetEvalAutomorphismKeyMapPtr(ciphertext->GetKeyTag());
  const auto key = detail::rotation_key(*keys, automorphism, context);
  const auto first = detail::pack(elements[0], source);
  return detail::rotation_result(ciphertext, detail::run_hybrid_core(
      elements[1], key, source, make_session, first, automorphism));
}

// The digit decomposition stays tied to this input and session for the whole
// batch. A zero shift follows OpenFHE's EvalFastRotation identity behavior.
template <class MakeSession>
std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> hybrid_rotate_many(
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly> &ciphertext,
    std::span<const int32_t> slot_shifts, MakeSession make_session) {
  proposal::require(ciphertext && ciphertext->GetCryptoContext(), "missing OpenFHE ciphertext context");
  const auto &elements = ciphertext->GetElements();
  proposal::require(elements.size() == 2, "Hybrid rotation requires a relinearized ciphertext");
  std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> results;
  results.reserve(slot_shifts.size());
  if (std::ranges::all_of(slot_shifts, [](int32_t shift) { return shift == 0; })) {
    for (size_t index = 0; index < slot_shifts.size(); ++index)
      results.push_back(ciphertext->Clone());
    return results;
  }

  const auto context = ciphertext->GetCryptoContext();
  const auto source = detail::source_layout(elements[1], context);
  const auto dimension = static_cast<int32_t>(source.parameters->n());
  const auto keys = context->GetEvalAutomorphismKeyMapPtr(ciphertext->GetKeyTag());
  struct Rotation {
    uint32_t automorphism = 0;
    lbcrypto::EvalKey<lbcrypto::DCRTPoly> key;
  };
  std::vector<Rotation> requests;
  requests.reserve(slot_shifts.size());
  for (const auto shift : slot_shifts) {
    if (shift == 0) {
      requests.emplace_back();
    } else {
      const auto automorphism = detail::rotation_index(context, shift, dimension);
      requests.push_back({automorphism, detail::rotation_key(*keys, automorphism, context)});
    }
  }

  const auto first = detail::pack(elements[0], source);
  const auto second = detail::pack(elements[1], source);
  auto session = make_session(detail::resident_base(source));
  const auto c0 = session.upload(source, first);
  const auto digits = session.hybrid_digits(session.upload(source, second));
  for (const auto &request : requests) {
    if (!request.key) {
      results.push_back(ciphertext->Clone());
      continue;
    }
    const auto key = detail::PackedHybridKey(request.key, source).upload(session);
    const auto extended = session.hybrid_accumulate(digits, key);
    const auto rotated = session.hybrid_rotate_extended(extended, c0, request.automorphism);
    results.push_back(detail::rotation_result(ciphertext, detail::reduce_and_download(
        session, rotated, source, elements[0].GetParams())));
  }
  session.close();
  return results;
}
} // namespace ntt::openfhe
