#include "helper.cuh"
#include "hmm.cuh"
#include "inferNew.cuh"

using namespace troy;
using namespace std;

void testMatMult(int m, int l, int n) {
    EncryptionParameters parms(SchemeType::CKKS);
    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::create(
        poly_modulus_degree, {60, 30, 30, 30, 30, 30, 30, 30, 60}));
    double scale = pow(2.0, 30);

    auto context = HeContext::create(parms, true, SecurityLevel::Classical128);
    print_parameters(*context);
    cout << endl;

    CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();

    if (utils::device_count() > 0) {
        context->to_device_inplace();
        encoder.to_device_inplace();
    }

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key = keygen.create_public_key(false);
    RelinKeys relin_keys = keygen.create_relin_keys(false);
    GaloisKeys galois_keys = keygen.create_galois_keys(false);
    Encryptor encryptor(context);
    encryptor.set_public_key(public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    vector<vector<double>> a = generate_rand_matrix<double>(m, l);
    vector<vector<double>> b = generate_rand_matrix<double>(l, n);
    vector<vector<double>> c = multiply_matrices(a, b);

    cout << "a: ";
    print_matrix(a);
    cout << "b: ";
    print_matrix(b);

    cout << "c: ";
    print_matrix(c);

    Ciphertext a_ct;
    Ciphertext b_ct;
    Ciphertext c_ct;

    if (l < n) {
        encode_matrix_l(a, n);
        vector<complex<double>> a_vec(slot_count, 0);
        pack_matrix(a, a_vec);
        Plaintext a_pt;
        encoder.encode_complex64_simd(a_vec, std::nullopt, scale, a_pt);

        encryptor.encrypt_asymmetric(a_pt, a_ct);

        vector<complex<double>> b_vec(slot_count, 0);
        pack_matrix(b, b_vec);
        Plaintext b_pt;
        encoder.encode_complex64_simd(b_vec, std::nullopt, scale, b_pt);
        encryptor.encrypt_asymmetric(b_pt, b_ct);

    } else {
        vector<complex<double>> a_vec(slot_count, 0);
        pack_matrix(a, a_vec);
        Plaintext a_pt;
        encoder.encode_complex64_simd(a_vec, std::nullopt, scale, a_pt);
        encryptor.encrypt_asymmetric(a_pt, a_ct);

        encode_matrix_r(b);
        vector<complex<double>> b_vec(slot_count, 0);
        pack_matrix(b, b_vec);
        Plaintext b_pt;
        encoder.encode_complex64_simd(b_vec, std::nullopt, scale, b_pt);
        encryptor.encrypt_asymmetric(b_pt, b_ct);
    }

    Hmm hmm(&encoder, &evaluator, relin_keys, galois_keys, scale);

    auto start = clock();
    hmm.matMult(a_ct, b_ct, c_ct, m, l, n);
    auto stop = clock();
    auto esp_time = (float)(stop - start) / CLOCKS_PER_SEC;

    Plaintext c_pt;
    decryptor.decrypt(c_ct, c_pt);
    vector<complex<double>> res;
    encoder.decode_complex64_simd(c_pt, res);
    auto matrix = convertTo2D(res, m, max(l, n));
    print_matrix(matrix);
    printf("The time by matrixMul:\t%fs\n", esp_time);
}

int main() {
    // int m, l, n;
    // while (cin >> m >> l >> n) {
    //     testMatMult(m, l, n);
    // }
    infer();
}
