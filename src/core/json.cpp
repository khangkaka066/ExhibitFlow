#include "exhibitflow/json.hpp"

#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace exhibitflow {

ContractError::ContractError(const std::string& message)
    : std::runtime_error(message) {}

JsonValue::JsonValue()
    : type_(Type::Null), bool_value_(false), number_value_(0.0) {}

JsonValue::JsonValue(bool value)
    : type_(Type::Bool), bool_value_(value), number_value_(0.0) {}

JsonValue::JsonValue(double value)
    : type_(Type::Number), bool_value_(false), number_value_(value) {}

JsonValue::JsonValue(std::string value)
    : type_(Type::String), bool_value_(false), number_value_(0.0), string_value_(std::move(value)) {}

JsonValue::JsonValue(Array value)
    : type_(Type::Array), bool_value_(false), number_value_(0.0), array_value_(std::move(value)) {}

JsonValue::JsonValue(Object value)
    : type_(Type::Object), bool_value_(false), number_value_(0.0), object_value_(std::move(value)) {}

JsonValue::Type JsonValue::type() const {
    return type_;
}

bool JsonValue::is_null() const {
    return type_ == Type::Null;
}

bool JsonValue::as_bool() const {
    if (type_ != Type::Bool) {
        throw ContractError("expected JSON boolean");
    }
    return bool_value_;
}

double JsonValue::as_number() const {
    if (type_ != Type::Number) {
        throw ContractError("expected JSON number");
    }
    return number_value_;
}

const std::string& JsonValue::as_string() const {
    if (type_ != Type::String) {
        throw ContractError("expected JSON string");
    }
    return string_value_;
}

const JsonValue::Array& JsonValue::as_array() const {
    if (type_ != Type::Array) {
        throw ContractError("expected JSON array");
    }
    return array_value_;
}

const JsonValue::Object& JsonValue::as_object() const {
    if (type_ != Type::Object) {
        throw ContractError("expected JSON object");
    }
    return object_value_;
}

const JsonValue* JsonValue::find(const std::string& key) const {
    if (type_ != Type::Object) {
        return nullptr;
    }

    const auto it = object_value_.find(key);
    if (it == object_value_.end()) {
        return nullptr;
    }

    return &it->second;
}

namespace {

class JsonParser {
public:
    explicit JsonParser(const std::string& source)
        : source_(source) {}

    JsonValue parse() {
        skip_whitespace();
        JsonValue value = parse_value();
        skip_whitespace();
        if (position_ != source_.size()) {
            fail("unexpected trailing characters");
        }
        return value;
    }

private:
    JsonValue parse_value() {
        skip_whitespace();
        if (position_ >= source_.size()) {
            fail("unexpected end of JSON");
        }

        const char next = source_[position_];
        if (next == '{') {
            return JsonValue(parse_object());
        }
        if (next == '[') {
            return JsonValue(parse_array());
        }
        if (next == '"') {
            return JsonValue(parse_string());
        }
        if (next == '-' || std::isdigit(static_cast<unsigned char>(next))) {
            return JsonValue(parse_number());
        }
        if (consume_literal("true")) {
            return JsonValue(true);
        }
        if (consume_literal("false")) {
            return JsonValue(false);
        }
        if (consume_literal("null")) {
            return JsonValue();
        }

        fail("unexpected JSON value");
    }

    JsonValue::Object parse_object() {
        expect('{');
        JsonValue::Object object;
        skip_whitespace();

        if (peek('}')) {
            expect('}');
            return object;
        }

        while (true) {
            skip_whitespace();
            if (!peek('"')) {
                fail("expected object key");
            }
            std::string key = parse_string();
            skip_whitespace();
            expect(':');
            object.emplace(std::move(key), parse_value());
            skip_whitespace();

            if (peek('}')) {
                expect('}');
                break;
            }
            expect(',');
        }

        return object;
    }

    JsonValue::Array parse_array() {
        expect('[');
        JsonValue::Array array;
        skip_whitespace();

        if (peek(']')) {
            expect(']');
            return array;
        }

        while (true) {
            array.push_back(parse_value());
            skip_whitespace();

            if (peek(']')) {
                expect(']');
                break;
            }
            expect(',');
        }

        return array;
    }

    std::string parse_string() {
        expect('"');
        std::string value;

        while (position_ < source_.size()) {
            const char ch = source_[position_++];
            if (ch == '"') {
                return value;
            }

            if (static_cast<unsigned char>(ch) < 0x20) {
                fail("control character in JSON string");
            }

            if (ch != '\\') {
                value.push_back(ch);
                continue;
            }

            if (position_ >= source_.size()) {
                fail("unterminated escape sequence");
            }

            const char escaped = source_[position_++];
            switch (escaped) {
                case '"':
                case '\\':
                case '/':
                    value.push_back(escaped);
                    break;
                case 'b':
                    value.push_back('\b');
                    break;
                case 'f':
                    value.push_back('\f');
                    break;
                case 'n':
                    value.push_back('\n');
                    break;
                case 'r':
                    value.push_back('\r');
                    break;
                case 't':
                    value.push_back('\t');
                    break;
                case 'u':
                    fail("unicode escapes are not supported in this skeleton parser");
                    break;
                default:
                    fail("invalid JSON escape sequence");
            }
        }

        fail("unterminated JSON string");
    }

    double parse_number() {
        const std::size_t start = position_;

        if (peek('-')) {
            ++position_;
        }

        if (peek('0')) {
            ++position_;
        } else {
            read_digits();
        }

        if (peek('.')) {
            ++position_;
            read_digits();
        }

        if (peek('e') || peek('E')) {
            ++position_;
            if (peek('+') || peek('-')) {
                ++position_;
            }
            read_digits();
        }

        try {
            return std::stod(source_.substr(start, position_ - start));
        } catch (const std::exception&) {
            fail("invalid JSON number");
        }
    }

    void read_digits() {
        const std::size_t start = position_;
        while (position_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[position_]))) {
            ++position_;
        }
        if (start == position_) {
            fail("expected digit");
        }
    }

    bool consume_literal(const char* literal) {
        const std::string text(literal);
        if (source_.compare(position_, text.size(), text) == 0) {
            position_ += text.size();
            return true;
        }
        return false;
    }

    bool peek(char expected) const {
        return position_ < source_.size() && source_[position_] == expected;
    }

    void expect(char expected) {
        if (!peek(expected)) {
            std::ostringstream message;
            message << "expected '" << expected << "'";
            fail(message.str());
        }
        ++position_;
    }

    void skip_whitespace() {
        while (position_ < source_.size() && std::isspace(static_cast<unsigned char>(source_[position_]))) {
            ++position_;
        }
    }

    [[noreturn]] void fail(const std::string& message) const {
        std::ostringstream detail;
        detail << message << " at byte " << position_;
        throw ContractError(detail.str());
    }

    const std::string& source_;
    std::size_t position_ = 0;
};

const JsonValue& required_field(const JsonValue& object, const std::string& key) {
    const JsonValue* value = object.find(key);
    if (value == nullptr) {
        throw ContractError("missing required field: " + key);
    }
    return *value;
}

std::int64_t read_int64(const JsonValue& object, const std::string& key) {
    const double value = required_field(object, key).as_number();
    if (!std::isfinite(value) || value < 0.0 || std::floor(value) != value) {
        throw ContractError("field must be a non-negative integer: " + key);
    }
    if (value > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
        throw ContractError("integer field is too large: " + key);
    }
    return static_cast<std::int64_t>(value);
}

double read_non_negative_number(const JsonValue& object, const std::string& key) {
    const double value = required_field(object, key).as_number();
    if (!std::isfinite(value) || value < 0.0) {
        throw ContractError("field must be a non-negative finite number: " + key);
    }
    return value;
}

std::string read_string(const JsonValue& object, const std::string& key) {
    const std::string value = required_field(object, key).as_string();
    if (value.empty()) {
        throw ContractError("field must be a non-empty string: " + key);
    }
    return value;
}

std::string escape_json_string(const std::string& value) {
    std::ostringstream out;
    for (const char ch : value) {
        switch (ch) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch)) << std::dec;
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

void append_bbox(std::ostringstream& out, const BoundingBox& bbox) {
    out << "[" << bbox.x << "," << bbox.y << "," << bbox.width << "," << bbox.height << "]";
}

}  // namespace

JsonValue parse_json(const std::string& source) {
    return JsonParser(source).parse();
}

DetectionFrame parse_detection_frame(const std::string& line, std::size_t line_number) {
    try {
        const JsonValue root = parse_json(line);
        const JsonValue::Object& object = root.as_object();

        DetectionFrame frame;
        frame.context.schema_version = read_string(root, "schema_version");
        if (frame.context.schema_version != "0.1") {
            throw ContractError("unsupported schema_version: " + frame.context.schema_version);
        }

        frame.context.sequence_id = read_string(root, "sequence_id");
        frame.context.camera_id = read_string(root, "camera_id");
        frame.context.frame_id = read_int64(root, "frame_id");
        frame.context.timestamp_ms = read_int64(root, "timestamp_ms");

        const JsonValue::Object& image_size = required_field(root, "image_size").as_object();
        frame.context.image_size.width = static_cast<int>(read_int64(JsonValue(image_size), "width"));
        frame.context.image_size.height = static_cast<int>(read_int64(JsonValue(image_size), "height"));
        if (frame.context.image_size.width <= 0 || frame.context.image_size.height <= 0) {
            throw ContractError("image_size width and height must be positive");
        }

        const JsonValue::Array& detections = required_field(root, "detections").as_array();
        frame.detections.reserve(detections.size());

        for (std::size_t i = 0; i < detections.size(); ++i) {
            const JsonValue::Object& detection_object = detections[i].as_object();
            const JsonValue::Array& bbox_values = required_field(detections[i], "bbox").as_array();
            if (bbox_values.size() != 4) {
                throw ContractError("bbox must contain exactly four numbers");
            }

            Detection detection;
            detection.bbox.x = bbox_values[0].as_number();
            detection.bbox.y = bbox_values[1].as_number();
            detection.bbox.width = bbox_values[2].as_number();
            detection.bbox.height = bbox_values[3].as_number();
            detection.score = read_non_negative_number(JsonValue(detection_object), "score");

            if (!std::isfinite(detection.bbox.x) || !std::isfinite(detection.bbox.y) ||
                !std::isfinite(detection.bbox.width) || !std::isfinite(detection.bbox.height)) {
                throw ContractError("bbox values must be finite numbers");
            }
            if (detection.bbox.x < 0.0 || detection.bbox.y < 0.0 ||
                detection.bbox.width <= 0.0 || detection.bbox.height <= 0.0) {
                throw ContractError("bbox must have non-negative origin and positive size");
            }
            if (detection.bbox.x + detection.bbox.width > frame.context.image_size.width ||
                detection.bbox.y + detection.bbox.height > frame.context.image_size.height) {
                throw ContractError("bbox must be inside image bounds");
            }
            if (detection.score > 1.0) {
                throw ContractError("score must be in range [0, 1]");
            }

            frame.detections.push_back(detection);
        }

        (void)object;
        return frame;
    } catch (const ContractError& error) {
        std::ostringstream message;
        message << "line " << line_number << ": " << error.what();
        throw ContractError(message.str());
    }
}

std::string serialize_detection_frame(const DetectionFrame& frame) {
    std::ostringstream out;
    out << std::setprecision(15);
    out << "{\"schema_version\":\"" << escape_json_string(frame.context.schema_version)
        << "\",\"sequence_id\":\"" << escape_json_string(frame.context.sequence_id)
        << "\",\"camera_id\":\"" << escape_json_string(frame.context.camera_id)
        << "\",\"frame_id\":" << frame.context.frame_id
        << ",\"timestamp_ms\":" << frame.context.timestamp_ms
        << ",\"image_size\":{\"width\":" << frame.context.image_size.width
        << ",\"height\":" << frame.context.image_size.height
        << "},\"detections\":[";
    for (std::size_t i = 0; i < frame.detections.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        const Detection& detection = frame.detections[i];
        out << "{\"bbox\":";
        append_bbox(out, detection.bbox);
        out << ",\"score\":" << detection.score << "}";
    }
    out << "]}";
    return out.str();
}

std::string serialize_track_frame(const TrackFrame& frame) {
    std::ostringstream out;
    out << std::setprecision(15);
    out << "{\"schema_version\":\"" << escape_json_string(frame.context.schema_version)
        << "\",\"sequence_id\":\"" << escape_json_string(frame.context.sequence_id)
        << "\",\"camera_id\":\"" << escape_json_string(frame.context.camera_id)
        << "\",\"frame_id\":" << frame.context.frame_id
        << ",\"timestamp_ms\":" << frame.context.timestamp_ms
        << ",\"image_size\":{\"width\":" << frame.context.image_size.width
        << ",\"height\":" << frame.context.image_size.height
        << "},\"tracks\":[";

    for (std::size_t i = 0; i < frame.tracks.size(); ++i) {
        const auto& track = frame.tracks[i];
        if (i > 0) {
            out << ",";
        }
        out << "{\"track_id\":" << track.track_id << ",\"bbox\":";
        append_bbox(out, track.bbox);
        out << ",\"point_image\":[" << track.point_image_x << "," << track.point_image_y
            << "],\"score\":" << track.score << "}";
    }

    out << "]}";
    return out.str();
}

std::string parse_tracker_name_from_config(const std::string& source) {
    const JsonValue root = parse_json(source);
    const std::string schema_version = read_string(root, "schema_version");
    if (schema_version != "0.1") {
        throw ContractError("unsupported config schema_version: " + schema_version);
    }
    return read_string(root, "tracker");
}

}  // namespace exhibitflow
