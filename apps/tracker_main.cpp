#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "exhibitflow/json.hpp"
#include "exhibitflow/tracker.hpp"

namespace {

constexpr const char* kVersion = "0.1.0";

struct CliOptions {
    std::string input_path;
    std::string config_path;
    std::string output_path;
    bool help = false;
    bool version = false;
};

void print_help(std::ostream& out) {
    out << "Usage: exhibitflow_tracker --input detections.jsonl --config config.json --output tracks.jsonl\n"
        << "\n"
        << "Options:\n"
        << "  --input PATH    Detection JSONL input file\n"
        << "  --config PATH   Tracker config JSON file\n"
        << "  --output PATH   Track JSONL output file\n"
        << "  --help          Show this help text\n"
        << "  --version       Show CLI version\n";
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            options.help = true;
        } else if (arg == "--version") {
            options.version = true;
        } else if (arg == "--input" || arg == "--config" || arg == "--output") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + arg);
            }
            const std::string value = argv[++i];
            if (arg == "--input") {
                options.input_path = value;
            } else if (arg == "--config") {
                options.config_path = value;
            } else {
                options.output_path = value;
            }
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (!options.help && !options.version) {
        if (options.input_path.empty() || options.config_path.empty() || options.output_path.empty()) {
            throw std::invalid_argument("--input, --config, and --output are required");
        }
        if (options.output_path == options.input_path || options.output_path == options.config_path) {
            throw std::invalid_argument("--output must be different from --input and --config");
        }
    }

    return options;
}

std::string read_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::ios_base::failure("could not open file: " + path);
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void validate_frame_order(
    const exhibitflow::DetectionFrame& frame,
    const exhibitflow::DetectionFrame* previous
) {
    if (previous == nullptr) {
        return;
    }

    if (frame.context.sequence_id != previous->context.sequence_id) {
        throw exhibitflow::ContractError("sequence_id changed inside one run");
    }
    if (frame.context.camera_id != previous->context.camera_id) {
        throw exhibitflow::ContractError("camera_id changed inside one run");
    }
    if (frame.context.frame_id <= previous->context.frame_id) {
        throw exhibitflow::ContractError("frame_id must be strictly increasing");
    }
    if (frame.context.timestamp_ms <= previous->context.timestamp_ms) {
        throw exhibitflow::ContractError("timestamp_ms must be strictly increasing");
    }
}

int run_tracker(const CliOptions& options) {
    namespace fs = std::filesystem;

    const fs::path output_path(options.output_path);
    if (fs::exists(output_path)) {
        throw std::ios_base::failure("output file already exists: " + options.output_path);
    }

    const std::string config_source = read_file(options.config_path);
    auto tracker = exhibitflow::create_tracker(exhibitflow::parse_tracker_name_from_config(config_source));

    std::ifstream input(options.input_path);
    if (!input) {
        throw std::ios_base::failure("could not open input file: " + options.input_path);
    }

    const fs::path temporary_path = output_path.string() + ".tmp";
    std::ofstream output(temporary_path);
    if (!output) {
        throw std::ios_base::failure("could not open output file: " + temporary_path.string());
    }

    std::string line;
    std::size_t line_number = 0;
    bool have_previous = false;
    exhibitflow::DetectionFrame previous;

    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }

        exhibitflow::DetectionFrame frame = exhibitflow::parse_detection_frame(line, line_number);
        validate_frame_order(frame, have_previous ? &previous : nullptr);

        const std::vector<exhibitflow::TrackObservation> tracks =
            tracker->update(frame.context, frame.detections);

        exhibitflow::TrackFrame track_frame;
        track_frame.context = frame.context;
        track_frame.tracks = tracks;
        output << exhibitflow::serialize_track_frame(track_frame) << "\n";

        previous = frame;
        have_previous = true;
    }

    output.close();
    if (!output) {
        throw std::ios_base::failure("failed while writing output file: " + temporary_path.string());
    }

    fs::rename(temporary_path, output_path);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);

        if (options.help) {
            print_help(std::cout);
            return 0;
        }

        if (options.version) {
            std::cout << "exhibitflow_tracker " << kVersion << "\n";
            return 0;
        }

        return run_tracker(options);
    } catch (const std::invalid_argument& error) {
        std::cerr << "argument error: " << error.what() << "\n";
        return 2;
    } catch (const exhibitflow::ContractError& error) {
        std::cerr << "contract error: " << error.what() << "\n";
        return 2;
    } catch (const std::ios_base::failure& error) {
        std::cerr << "io error: " << error.what() << "\n";
        return 3;
    } catch (const std::exception& error) {
        std::cerr << "runtime error: " << error.what() << "\n";
        return 4;
    }
}
