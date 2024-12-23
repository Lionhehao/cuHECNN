#include "helper.cuh"

class Hmm {
public:
    Hmm(troy::CKKSEncoder* encoder, troy::Evaluator* evaluator,
        troy::RelinKeys relin_keys, troy::GaloisKeys galois_keys, size_t scale);

    void matMult(troy::Ciphertext a, troy::Ciphertext b, troy::Ciphertext& c,
                 int m, int l, int n);

private:
    troy::CKKSEncoder* encoder;
    troy::Evaluator* evaluator;
    troy::RelinKeys relin_keys;
    troy::GaloisKeys galois_keys;
    size_t scale;
    size_t slot_count;
};