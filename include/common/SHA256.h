#ifndef GPU_ANALYZER_SHA256_H
#define GPU_ANALYZER_SHA256_H

#include <cstdint>
#include <string>

// Minimal SHA-256 implementation (no external dependency).
// Used for fingerprinting trace files to validate the SoA binary cache.

// Compute SHA-256 of a file's contents. Writes 32 bytes into out[].
// Sets out to all-zeros on failure and prints a warning.
void sha256_file(const std::string &file_path, uint8_t out[32]);

// Compute SHA-256 of an in-memory buffer. Writes 32 bytes into out[].
void sha256_buf(const uint8_t *data, size_t len, uint8_t out[32]);

#endif // GPU_ANALYZER_SHA256_H
