#pragma once

#include <stdexcept>
#include <string>

namespace engine::runtime {

// A request the device cannot serve AT THIS SIZE -- e.g. a transcription
// prompt plus audio whose prefill graph does not fit in VRAM. Distinct from a
// genuine internal fault: the caller can fix it by sending less, so servers
// should surface it as a client error rather than an opaque 500.
class CapacityError : public std::runtime_error {
public:
    explicit CapacityError(const std::string & message) : std::runtime_error(message) {}
};

// A request the session turns away as sent: an input or option it does not
// take, or one that does not read, such as a malformed input artifact. The
// caller can fix it, so servers answer it as a client error, as they do
// CapacityError. A fault of the model or the server stays a plain
// runtime_error.
class InvalidRequestError : public std::runtime_error {
public:
    explicit InvalidRequestError(const std::string & message) : std::runtime_error(message) {}
};

}  // namespace engine::runtime
