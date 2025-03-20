#include "hmm.cuh"
#include "inferNew.cuh"
using namespace std;
using namespace troy;

void infer() {
    //////////////// Scheme Generate //////////////////
    cout << "Scheme Generate" << endl;

    EncryptionParameters parms(SchemeType::CKKS);
    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::create(
        poly_modulus_degree, {45, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 45}));
    double scale = pow(2.0, 34);

    auto context = HeContext::create(parms, true, SecurityLevel::Classical128);
    print_parameters(*context);
    cout << endl;

    CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();

    if (utils::device_count() > 0) {
        context->to_device_inplace();
        encoder.to_device_inplace();
    }

    auto keyGen_start = clock();

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key = keygen.create_public_key(false);
    RelinKeys relin_keys = keygen.create_relin_keys(false);
    GaloisKeys galois_keys = keygen.create_galois_keys(false);

    auto keyGen_end = clock();
    cout << "keyGen: " << (float)(keyGen_end - keyGen_start) / CLOCKS_PER_SEC
         << "s" << endl;
    
    Encryptor encryptor(context);
    encryptor.set_public_key(public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    Hmm hmm(&encoder, &evaluator, relin_keys, galois_keys, scale);
    //////////////// Encrypt Model //////////////////
    cout << "Encrypt Model" << endl;
    string weightfile = "../data/model_weights.csv";
    auto weights = readCsvToWeights(weightfile);

    auto modelEnc_start = clock();
    vector<Ciphertext> conv_cts;
    vector<Ciphertext> conv_b_cts;
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 49; j++) {
            vector<complex<double>> conv(slot_count, 0);
            for (size_t k = 0; k < slot_count; k++) {
                conv[k] = weights[0][i * 49 + j];
            }
            Plaintext conv_pt;
            encoder.encode_complex64_simd(conv, std::nullopt, scale, conv_pt);
            Ciphertext conv_ct;
            encryptor.encrypt_asymmetric(conv_pt, conv_ct);
            conv_cts.push_back(conv_ct);
        }

        vector<complex<double>> conv_b(slot_count, 0);
        for (size_t j = 0; j < slot_count; j++) {
            conv_b[j] = weights[1][i];
        }
        Plaintext conv_b_pt;
        encoder.encode_complex64_simd(conv_b, std::nullopt, scale, conv_b_pt);
        Ciphertext conv_b_ct;
        encryptor.encrypt_asymmetric(conv_b_pt, conv_b_ct);
        conv_b_cts.push_back(conv_b_ct);
    }

    vector<Ciphertext> fct1_cts;
    Ciphertext fct1_b_ct;

    for (size_t i = 0; i < 256; i++) {
        vector<complex<double>> fct1(slot_count, 0);
        for (size_t j = 0; j < 64; j++) {
            fct1[j] = weights[2][i + j * 256];
        }
        Plaintext fct1_pt;
        encoder.encode_complex64_simd(fct1, std::nullopt, scale, fct1_pt);
        Ciphertext fct1_ct;
        encryptor.encrypt_asymmetric(fct1_pt, fct1_ct);
        fct1_cts.push_back(fct1_ct);
    }

    vector<complex<double>> fct1_b(slot_count, 0);
    for (size_t i = 0; i < 128; i++) {
        copy(weights[3].begin(), weights[3].end(), fct1_b.begin() + i * 64);
    }
    Plaintext fct1_b_pt;
    encoder.encode_complex64_simd(fct1_b, std::nullopt, scale, fct1_b_pt);
    encryptor.encrypt_asymmetric(fct1_b_pt, fct1_b_ct);

    vector<Ciphertext> fct2_cts;
    Ciphertext fct2_b_ct;

    for (size_t i = 0; i < 10; i++) {
        vector<complex<double>> fct2(slot_count, 0);
        for (size_t j = 0; j < 64; j++) {
            fct2[j] = weights[4][i * 64 + j];
        }
        Plaintext fct2_pt;
        encoder.encode_complex64_simd(fct2, std::nullopt, scale, fct2_pt);
        Ciphertext fct2_ct;
        encryptor.encrypt_asymmetric(fct2_pt, fct2_ct);
        fct2_cts.push_back(fct2_ct);
    }

    vector<complex<double>> fct2_b(slot_count, 0);
    for (size_t i = 0; i < 10; i++) {
        copy(weights[5].begin(), weights[5].end(), fct2_b.begin() + i * 64);
    }
    Plaintext fct2_b_pt;
    encoder.encode_complex64_simd(fct2_b, std::nullopt, scale, fct2_b_pt);
    encryptor.encrypt_asymmetric(fct2_b_pt, fct2_b_ct);

    auto modelEnc_end = clock();
    cout << "modelEnc: " << (float)(modelEnc_end - modelEnc_start) / CLOCKS_PER_SEC
         << "s" << endl;

    //////////////// Encrypt Data //////////////////
    cout << "Encrypt Data" << endl;
    string testfile = "../data/MNISTt10k(28x28).csv";
    auto test_datas = readCsvToTestDatas(testfile);

    auto dataEnc_start = clock();

    vector<Ciphertext> test_data_cts;
    for (size_t i = 0; i < 49; i++) {
        vector<complex<double>> test_data(slot_count, 0);
        size_t index = (i / 7) * 28 + (i % 7);
        for (size_t j = 0; j < 128; j++) {
            for (size_t k = 0; k < 64; k++) {
                test_data[j * 64 + k] =
                    (test_datas[j][index + (k / 8) * 3 * 28 + (k % 8) * 3 + 1] /
                         255.0 -
                     0.1307) /
                    0.3081;
            }
        }

        Plaintext test_data_pt;
        encoder.encode_complex64_simd(test_data, std::nullopt, scale,
                                      test_data_pt);
        Ciphertext test_data_ct;
        encryptor.encrypt_asymmetric(test_data_pt, test_data_ct);
        test_data_cts.push_back(test_data_ct);
    }

    auto dataEnc_end = clock();
    cout << "dataEnc: " << (float)(dataEnc_end - dataEnc_start) / CLOCKS_PER_SEC
         << "s" << endl;

    //////////////// Infer //////////////////
    cout << "Infer" << endl;

    auto infer_start = clock();

    vector<Ciphertext> conv_reses;
    for (size_t i = 0; i < 4; i++) {
        Ciphertext conv_res;
        for (size_t j = 0; j < 49; j++) {
            Ciphertext tmp;
            evaluator.multiply(test_data_cts[j], conv_cts[i * 49 + j], tmp);
            evaluator.relinearize_inplace(tmp, relin_keys);
            evaluator.rescale_to_next_inplace(tmp);
            if (j == 0) {
                conv_res = tmp;
            } else {
                evaluator.add_inplace(conv_res, tmp);
            }
        }
        conv_b_cts[i].scale() = scale;
        conv_res.scale() = scale;
        evaluator.mod_switch_to_inplace(conv_b_cts[i], conv_res.parms_id());
        evaluator.add_inplace(conv_res, conv_b_cts[i]);
        conv_reses.push_back(conv_res);
    }

    auto act1_start = clock();
    cout << "conv: " << (float)(act1_start - infer_start) / CLOCKS_PER_SEC
         << "s" << endl;

    for (size_t i = 0; i < 4; i++) {
        evaluator.square_inplace(conv_reses[i]);
        evaluator.relinearize_inplace(conv_reses[i], relin_keys);
        evaluator.rescale_to_next_inplace(conv_reses[i]);
    }

    auto fct1_start = clock();
    cout << "act1: " << (float)(fct1_start - act1_start) / CLOCKS_PER_SEC << "s"
         << endl;

    Ciphertext fct1_res;
    for (size_t i = 0; i < 256; i++) {
        vector<complex<double>> a_mask(slot_count, 0);
        for (size_t j = i % 64; j < 128 * 64; j += 64) {
            a_mask[j] = 1;
        }
        Plaintext a_mask_pt;
        encoder.encode_complex64_simd(a_mask, conv_reses[i / 64].parms_id(),
                                      scale, a_mask_pt);

        Ciphertext a;
        evaluator.multiply_plain(conv_reses[i / 64], a_mask_pt, a);
        evaluator.rescale_to_next_inplace(a);

        int step = i % 64;
        if (step != 0) {
            evaluator.rotate_vector_inplace(a, step, galois_keys);
        }

        evaluator.mod_switch_to_inplace(fct1_cts[i], a.parms_id());
        Ciphertext tmp;

        hmm.matMult(a, fct1_cts[i], tmp, 128, 1, 64);

        if (i == 0) {
            fct1_res = tmp;
        } else {
            evaluator.add_inplace(fct1_res, tmp);
        }
    }

    fct1_b_ct.scale() = fct1_res.scale();
    evaluator.mod_switch_to_inplace(fct1_b_ct, fct1_res.parms_id());
    evaluator.add_inplace(fct1_res, fct1_b_ct);

    auto act2_start = clock();
    cout << "fct1: " << (float)(act2_start - fct1_start) / CLOCKS_PER_SEC << "s"
         << endl;

    evaluator.square_inplace(fct1_res);
    evaluator.relinearize_inplace(fct1_res, relin_keys);
    evaluator.rescale_to_next_inplace(fct1_res);

    auto fct2_start = clock();
    cout << "act2: " << (float)(fct2_start - act2_start) / CLOCKS_PER_SEC << "s"
         << endl;

    Ciphertext fct2_res;
    for (size_t i = 0; i < 10; i++) {
        Ciphertext tmp;
        fct2_cts[i].scale() = fct1_res.scale();
        evaluator.mod_switch_to_inplace(fct2_cts[i], fct1_res.parms_id());
        hmm.matMult(fct1_res, fct2_cts[i], tmp, 128, 64, 1);

        if (i == 0) {
            fct2_res = tmp;
        } else {
            evaluator.rotate_vector_inplace(tmp, -i, galois_keys);
            evaluator.add_inplace(fct2_res, tmp);
        }
    }

    fct2_b_ct.scale() = fct2_res.scale();
    evaluator.mod_switch_to_inplace(fct2_b_ct, fct2_res.parms_id());
    evaluator.add_inplace(fct2_res, fct2_b_ct);

    auto infer_stop = clock();
    cout << "fct2: " << (float)(infer_stop - fct2_start) / CLOCKS_PER_SEC << "s"
         << endl;
    auto esp_time = (float)(infer_stop - infer_start) / CLOCKS_PER_SEC;
    printf("The time by matrixMul:\t%fs\n", esp_time);

    Plaintext res_pt;
    decryptor.decrypt(fct2_res, res_pt);
    vector<complex<double>> res;
    encoder.decode_complex64_simd(res_pt, res);
    cout << "batch result:";
    for (size_t i = 0; i < 128; i++) {
        double ans = res[i * 64].real();
        int ans_index = 0;
        for (size_t j = 1; j < 10; j++) {
            if (res[i * 64 + j].real() > ans) {
                ans = res[i * 64 + j].real();
                ans_index = j;
            }
        }
        if (i % 32 == 0) {
            cout << endl;
        }
        cout << " " << ans_index;
    }
    cout << endl;
}
