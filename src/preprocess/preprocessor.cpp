// This preprocessor is adapted from paq8l, paq8hp12any and paq8px.

#include <vector>
#include <cstdlib>
#include <string.h>

#include "preprocessor.h"
#include "dictionary.h"

namespace preprocessor {

bool IsAscii(int byte) {
  if (byte >= 9 && byte <= 13) return true;
  if (byte >= 32 && byte <= 126) return true;
  return false;
}

void Pretrain(Predictor* p, FILE* dictionary) {
  if (dictionary == NULL) return;
  fseek(dictionary, 0L, SEEK_END);
  unsigned int len = ftell(dictionary);
  fseek(dictionary, 0L, SEEK_SET);

  std::vector<unsigned char> header;
  header.push_back(DEFAULT);
  header.push_back(len>>24);
  header.push_back(len>>16);
  header.push_back(len>>8);
  header.push_back(len);

  for (unsigned int i = 0; i < header.size(); ++i) {
    for (int j = 7; j >= 0; --j) {
      p->Pretrain((header[i]>>j)&1);
    }
  }

  unsigned int percent = 1 + (len / 10000);
  for (unsigned int i = 0; i < len; ++i) {
    unsigned char c = getc(dictionary);
    if (c == '\n') c = ' ';
    if (i % percent == 0) {
      double frac = 100.0 * i / len;
      fprintf(stderr, "\rpretraining: %.2f%%", frac);
      fflush(stderr);
    }
    for (int j = 7; j >= 0; --j) {
      p->Pretrain((c>>j)&1);
    }
  }
}

Filetype detect(FILE* in, int n, Filetype type) {
  long start=ftell(in);

  // For TEXT detection
  int ascii_start = -1;
  int ascii_run = 0;
  int space_count = 0;

  for (int i=0; i<n; ++i) {
    int c=getc(in);
    if (c==EOF) return (Filetype)(-1);

    // Detect TEXT
    if (type == DEFAULT) {
      if (IsAscii(c)) {
        if (ascii_start == -1) {
          ascii_start = i;
          ascii_run = 0;
          space_count = 0;
        }
        if (c == ' ') ++space_count;
        ++ascii_run;
        if (ascii_run > 500) {
          if (space_count < 5) {
            ascii_start = -1;
          } else {
            return fseek(in, start + ascii_start, SEEK_SET), TEXT;
          }
        }
      } else {
        ascii_start = -1;
      }
    } else if (type == TEXT) {
      if (IsAscii(c)) {
        ascii_run -= 2;
        if (ascii_run < 0) ascii_run = 0;
      } else {
        ascii_run += 3;
        if (ascii_run > 300) {
          return fseek(in, ftell(in) - 100, SEEK_SET), DEFAULT;
        }
      }
    }
  }
  return type;
}

void encode_default(FILE* in, FILE* out, int len) {
  while (len--) putc(getc(in), out);
}

int decode_default(FILE* in) {
  return getc(in);
}

void encode_text(FILE* in, FILE* out, int len, std::string temp_path,
    FILE* dictionary) {
  if (dictionary == NULL) {
    putc(0, out);
    for (int i = 0; i < len; ++i) {
      putc(getc(in), out);
    }
    return;
  }
  std::string path = temp_path + "2";
  FILE* temp_output = fopen(path.c_str(), "wb+");
  if (!temp_output) abort();
  int orig_pos = ftell(in);

  Dictionary dict(dictionary, true, false);
  dict.Encode(in, len, temp_output);

  int size = ftell(temp_output);
  if (size > len - 50) {
    putc(0, out);
    fseek(in, orig_pos, SEEK_SET);
    for (int i = 0; i < len; ++i) {
      putc(getc(in), out);
    }
  } else {
    putc(1, out);
    rewind(temp_output);
    for (int i = 0; i < size; ++i) {
      int c = getc(temp_output);
      if (c>='{' && c<127) c+='P'-'{';
      else if (c>='P' && c<'T') c-='P'-'{';
      else if ( (c>=':' && c<='?') || (c>='J' && c<='O') ) c^=0x70;
      if (c=='X' || c=='`') c^='X'^'`';
      putc(c, out);
    }
  }

  fclose(temp_output);
  remove(path.c_str());
}

Dictionary* dict = NULL;
bool wrt_enabled = true;

void reset_text_decoder(FILE* in, FILE* dictionary) {
  int c = getc(in);
  if (c) {
    wrt_enabled = true;
    if (dict == NULL) dict = new Dictionary(dictionary, false, true);
  } else {
    wrt_enabled = false;
  }
}

int decode_text(FILE* in) {
  if (!wrt_enabled) return getc(in);
  return dict->Decode(in);
}


void EncodeSegment(FILE* in, FILE* out, int n, const std::string& temp_path,
    FILE* dictionary, std::vector<double>* block_stats) {
  Filetype type=DEFAULT;
  long begin=ftell(in);

  long start = begin;
  int remainder = n;
  while (remainder > 0) {
    Filetype nextType=detect(in, remainder, type);
    long end=ftell(in);
    int len=int(end-begin);
    switch (type) {
      case TEXT: (*block_stats)[0] += len; break;
      case DEFAULT:
      default: (*block_stats)[1] += len; break;
    }
    remainder-=len;
    type=nextType;
    begin=end;
  }
  fseek(in, start, SEEK_SET);
  type = DEFAULT;
  begin = start;

  if ((*block_stats)[0] / n > 0.95) {
    (*block_stats)[0] = n;
    for (unsigned int i = 1; i < block_stats->size(); ++i) (*block_stats)[i] = 0;
    fprintf(out, "%c%c%c%c%c", TEXT, n>>24, n>>16, n>>8, n);
    encode_text(in, out, n, temp_path, dictionary);
    return;
  }

  while (n>0) {
    Filetype nextType=detect(in, n, type);
    long end=ftell(in);
    fseek(in, begin, SEEK_SET);
    int len=int(end-begin);
    if (len>0) {
      fprintf(out, "%c%c%c%c%c", type, len>>24, len>>16, len>>8, len);
      switch(type) {
        default: {
          encode_default(in, out, len); break;
        }
      }
    }
    n-=len;
    type=nextType;
    begin=end;
  }
}

const unsigned long long kMaxSegment = 0x80000000 - 1;

void Encode(FILE* in, FILE* out, unsigned long long n, const std::string&
    temp_path, FILE* dictionary) {
  std::vector<double> block_stats(2);
  unsigned long long size = n;
  while(n > 0) {
    int segment = n;
    if (n > kMaxSegment) segment = kMaxSegment;
    std::vector<double> segment_stats(2);
    EncodeSegment(in, out, segment, temp_path, dictionary, &segment_stats);
    for (int i = 0; i < 2; ++i) block_stats[i] += segment_stats[i];
    n -= segment;
  }
  for (int i = 0; i < 2; ++i) block_stats[i] /= size;
  printf("\rDetected block types:");
  if (block_stats[0] > 0) printf(" TEXT: %.1f%%", block_stats[0] * 100);
  if (block_stats[1] > 0) printf(" DEFAULT: %.1f%%", block_stats[1] * 100);
  printf("\n");
}

void NoPreprocess(FILE* in, FILE* out, unsigned long long n) {
  while(n > 0) {
    int segment = n;
    if (n > kMaxSegment) segment = kMaxSegment;
    fprintf(out, "%c%c%c%c%c", DEFAULT, segment>>24, segment>>16, segment>>8,
        segment);
    encode_default(in, out, segment);
    n -= segment;
  }
}

int DecodeByte(FILE* in, FILE* dictionary) {
  static Filetype type=DEFAULT;
  static int len=0, reset=0;
  while (len==0) {
    int c = getc(in);
    if (c == EOF) return -1;
    reset=1;
    type=(Filetype)c;
    len=getc(in)<<24;
    len|=getc(in)<<16;
    len|=getc(in)<<8;
    len|=getc(in);
    if (len<0) len=1;
    if (type == TEXT) reset_text_decoder(in, dictionary);
  }
  --len;
  switch (type) {
    case TEXT:    return decode_text(in);
    default: {
      return decode_default(in);
    }
  }
}

void Decode(FILE* in, FILE* out, FILE* dictionary) {
  while (true) {
    int result = DecodeByte(in, dictionary);
    if (result == -1) {
      if (dict != NULL) delete dict;
      return;
    }
    putc(result, out);
  }
}

}
