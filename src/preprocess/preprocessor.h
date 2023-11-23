#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <stdio.h>
#include <string>

#include "../predictor.h"

namespace preprocessor {

typedef enum {DEFAULT, TEXT} Filetype;

void Encode(FILE* in, FILE* out, unsigned long long n, const std::string&
    temp_path, FILE* dictionary);

void NoPreprocess(FILE* in, FILE* out, unsigned long long n);

void Pretrain(Predictor* p, FILE* dictionary);

void Decode(FILE* in, FILE* out, FILE* dictionary);

}

#endif
