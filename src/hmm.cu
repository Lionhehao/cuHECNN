#include "hmm.cuh"

using namespace troy;
using namespace std;

Hmm::Hmm(CKKSEncoder* encoder, Evaluator* evaluator, RelinKeys relin_keys,
         GaloisKeys galois_keys, size_t scale)
    : encoder(encoder),
      evaluator(evaluator),
      relin_keys(relin_keys),
      galois_keys(galois_keys),
      scale(scale) {
    slot_count = encoder->slot_count();
}

void Hmm::matMult(Ciphertext a, Ciphertext b, Ciphertext& c, int m, int l,
                  int n) {
    if (l < n) {

        Ciphertext a_;
        for (size_t i = 0; i < l; i++) {
            Plaintext mask_pt;
            vector<complex<double>> mask(slot_count, 0);
            generate_lmask(i, mask, m, n);
            encoder->encode_complex64_simd(mask, a.parms_id(), scale, mask_pt);

            if (i == 0) {
                evaluator->multiply_plain(a, mask_pt, a_);
                evaluator->rescale_to_next_inplace(a_);
            } else {
                Ciphertext tmp;
                evaluator->multiply_plain(a, mask_pt, tmp);
                evaluator->rescale_to_next_inplace(tmp);
                int step = i - m * n * i;
                if (abs(step) >= slot_count / 2) {
                    step = step > 0 ? step - slot_count : slot_count + step;
                }
                if (step != 0) {
                    evaluator->rotate_vector_inplace(tmp, step, galois_keys);
                }
                evaluator->add_inplace(a_, tmp);
            }
        }

        for (size_t i = 0; i < log2(n); i++) {
            Ciphertext tmp;
            evaluator->rotate_vector(a_, -pow(2, i), galois_keys, tmp);
            evaluator->add_inplace(a_, tmp);
        }
        
        Ciphertext b_;
        for (size_t i = 0; i < l; i++) {
            Plaintext mask_pt;
            vector<complex<double>> mask(slot_count, 0);
            generate_rmask(i, mask, m, l, n);
            encoder->encode_complex64_simd(mask, b.parms_id(), scale, mask_pt);
            if (i == 0) {
                evaluator->multiply_plain(b, mask_pt, b_);
                evaluator->rescale_to_next_inplace(b_);
            } else {
                Ciphertext tmp;
                evaluator->multiply_plain(b, mask_pt, tmp);
                evaluator->rescale_to_next_inplace(tmp);
                int step = n * i - m * n * i;
                if (abs(step) > slot_count / 2) {
                    step = step > 0 ? step - slot_count : slot_count + step;
                }
                if (step != 0) {
                    evaluator->rotate_vector_inplace(tmp, step, galois_keys);
                }
                evaluator->add_inplace(b_, tmp);
            }
        }

        for (size_t i = 0; i < log2(m); i++) {
            Ciphertext tmp;
            evaluator->rotate_vector(b_, -pow(2, i) * n, galois_keys, tmp);
            evaluator->add_inplace(b_, tmp);
        }
        
        evaluator->multiply(a_, b_, c);
        evaluator->relinearize_inplace(c, relin_keys);
        evaluator->rescale_to_next_inplace(c);

        for (size_t i = 0; i < log2(l); i++) {
            Ciphertext tmp;
            int step = pow(2, i) * m * n;
            if (abs(step) >= slot_count / 2) {
                step = step > 0 ? step - slot_count : slot_count + step;
            }
            evaluator->rotate_vector(c, step, galois_keys, tmp);
            evaluator->add_inplace(c, tmp);
        }

    } else {
        
        for (size_t i = 0; i < log2(n); i++) {
            Ciphertext tmp;
            int step = -pow(2, i) * m * l;
            if (abs(step) >= slot_count / 2) {
                step = step > 0 ? step - slot_count : slot_count + step;
            }
            evaluator->rotate_vector(a, step, galois_keys, tmp);
            evaluator->add_inplace(a, tmp);
        }
        
        size_t cur_n = n;
        if (n < m) {
            while (m - cur_n >= n) {
                Ciphertext tmp;
                int step = -cur_n * l;
                if (abs(step) >= slot_count / 2) {
                    step = step > 0 ? step - slot_count : slot_count + step;
                }
                evaluator->rotate_vector(b, step, galois_keys, tmp);
                evaluator->add_inplace(b, tmp);
                cur_n *= 2;
            }
        }

        Ciphertext b_;
        for (size_t i = 0; i < n; i++) {
            vector<complex<double>> umask(slot_count, 0);
            generate_mask(umask, i, min(cur_n, i + m), l);
            Plaintext umask_pt;
            encoder->encode_complex64_simd(umask, b.parms_id(), scale,
                                           umask_pt);
            Ciphertext u_ct;
            evaluator->multiply_plain(b, umask_pt, u_ct);
            evaluator->rescale_to_next_inplace(u_ct);

            if (i == 0) {
                b_ = u_ct;
            } else {
                int step = -i * m * l + i * l;
                if (abs(step) >= slot_count / 2) {
                    step = step > 0 ? step - slot_count : slot_count + step;
                }
                if (step != 0) {
                    evaluator->rotate_vector_inplace(u_ct, step, galois_keys);
                }
                evaluator->add_inplace(b_, u_ct);
            }
            if (i + m > cur_n) {
                vector<complex<double>> vmask(slot_count, 0);
                generate_mask(vmask, 0, m + i - cur_n, l);
                Plaintext vmask_pt;
                encoder->encode_complex64_simd(vmask, std::nullopt, scale,
                                               vmask_pt);
                Ciphertext v_ct;
                evaluator->multiply_plain(b, vmask_pt, v_ct);
                evaluator->rescale_to_next_inplace(v_ct);
                int step = -i * m * l + i * l - cur_n * l;
                if (abs(step) >= slot_count / 2) {
                    step = step > 0 ? step - slot_count : slot_count + step;
                }
                if (step != 0) {
                    evaluator->rotate_vector_inplace(v_ct, step, galois_keys);
                }
                evaluator->add_inplace(b_, v_ct);
            }
        }
        
        a.scale() = scale;
        b_.scale() = scale;
        evaluator->mod_switch_to_inplace(a, b_.parms_id());
        evaluator->multiply(a, b_, c);
        evaluator->relinearize_inplace(c, relin_keys);
        evaluator->rescale_to_next_inplace(c);

        for (size_t i = 0; i < log2(l); i++) {
            Ciphertext tmp;
            evaluator->rotate_vector(c, pow(2, i), galois_keys, tmp);
            evaluator->add_inplace(c, tmp);
        }
        
        vector<complex<double>> mask(slot_count, 0);
        generate_lmask(0, mask, m * n, l);
        Plaintext lmask_pt;
        encoder->encode_complex64_simd(mask, c.parms_id(), scale, lmask_pt);
        evaluator->multiply_plain_inplace(c, lmask_pt);
        evaluator->rescale_to_next_inplace(c);

        for (size_t i = 0; i < log2(n); i++) {
            Ciphertext tmp;
            evaluator->rotate_vector(c, -pow(2, i), galois_keys, tmp);
            evaluator->add_inplace(c, tmp);
        }
        
        vector<complex<double>> filter(slot_count, 0);
        generate_filter(filter, m, n, l);
        Plaintext filter_pt;
        encoder->encode_complex64_simd(filter, c.parms_id(), scale, filter_pt);
        evaluator->multiply_plain_inplace(c, filter_pt);
        evaluator->rescale_to_next_inplace(c);
        
        for (size_t i = 0; i < log2(n); i++) {
            Ciphertext tmp;
            int step = pow(2, i) * m * l;
            if (abs(step) >= slot_count / 2) {
                step = step > 0 ? step - slot_count : slot_count + step;
            }
            evaluator->rotate_vector(c, step, galois_keys, tmp);
            evaluator->add_inplace(c, tmp);
        }
    }
}
