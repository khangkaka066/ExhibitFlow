#pragma once

#include <iosfwd>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "exhibitflow/types.hpp"

namespace exhibitflow {

class ContractError : public std::runtime_error {
public:
    explicit ContractError(const std::string& message);
};

class JsonValue {
public:
    enum class Type {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object
    };

    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue>;

    JsonValue();
    explicit JsonValue(bool value);
    explicit JsonValue(double value);
    explicit JsonValue(std::string value);
    explicit JsonValue(Array value);
    explicit JsonValue(Object value);

    Type type() const;
    bool is_null() const;
    bool as_bool() const;
    double as_number() const;
    const std::string& as_string() const;
    const Array& as_array() const;
    const Object& as_object() const;

    const JsonValue* find(const std::string& key) const;

private:
    Type type_;
    bool bool_value_;
    double number_value_;
    std::string string_value_;
    Array array_value_;
    Object object_value_;
};

JsonValue parse_json(const std::string& source);
DetectionFrame parse_detection_frame(const std::string& line, std::size_t line_number);
std::string serialize_detection_frame(const DetectionFrame& frame);
std::string serialize_track_frame(const TrackFrame& frame);
std::string parse_tracker_name_from_config(const std::string& source);

}  // namespace exhibitflow
