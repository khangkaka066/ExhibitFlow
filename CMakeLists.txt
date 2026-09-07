cmake_minimum_required(VERSION 3.20)

project(ExhibitFlowTracker
    VERSION 0.1.0
    DESCRIPTION "ExhibitFlow tracker interface and CLI skeleton"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter)
endif()

add_library(exhibitflow_core
    src/core/json.cpp
    src/core/track_writer.cpp
    src/trackers/mock_tracker.cpp
)

target_include_directories(exhibitflow_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(exhibitflow_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

add_executable(exhibitflow_tracker
    apps/tracker_main.cpp
)

target_link_libraries(exhibitflow_tracker PRIVATE exhibitflow_core)

target_compile_options(exhibitflow_tracker PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

set_target_properties(exhibitflow_tracker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

if(BUILD_TESTING)
    add_executable(exhibitflow_json_tests
        tests/test_json.cpp
    )

    target_link_libraries(exhibitflow_json_tests PRIVATE exhibitflow_core)

    target_compile_options(exhibitflow_json_tests PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    )

    add_test(NAME exhibitflow_json_tests COMMAND exhibitflow_json_tests)

    add_executable(exhibitflow_mock_tracker_tests
        tests/test_mock_tracker.cpp
    )

    target_link_libraries(exhibitflow_mock_tracker_tests PRIVATE exhibitflow_core)

    target_compile_options(exhibitflow_mock_tracker_tests PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    )

    add_test(NAME exhibitflow_mock_tracker_tests COMMAND exhibitflow_mock_tracker_tests)

    add_test(
        NAME exhibitflow_cli_cleanup
        COMMAND ${CMAKE_COMMAND} -E rm -f
            ${CMAKE_BINARY_DIR}/tracks.jsonl
            ${CMAKE_BINARY_DIR}/tracks.jsonl.tmp
    )

    set_tests_properties(exhibitflow_cli_cleanup PROPERTIES
        FIXTURES_SETUP tracker_clean
    )

    add_test(
        NAME exhibitflow_cli_fixture
        COMMAND exhibitflow_tracker
            --input ${PROJECT_SOURCE_DIR}/examples/detections.jsonl
            --config ${PROJECT_SOURCE_DIR}/configs/mock.json
            --output ${CMAKE_BINARY_DIR}/tracks.jsonl
    )

    set_tests_properties(exhibitflow_cli_fixture PROPERTIES
        FIXTURES_REQUIRED tracker_clean
        FIXTURES_SETUP tracker_output
    )

    add_test(
        NAME exhibitflow_cli_expected_output
        COMMAND ${CMAKE_COMMAND} -E compare_files
            ${CMAKE_BINARY_DIR}/tracks.jsonl
            ${PROJECT_SOURCE_DIR}/examples/expected_tracks.jsonl
    )

    set_tests_properties(exhibitflow_cli_expected_output PROPERTIES
        FIXTURES_REQUIRED tracker_output
    )

    if(Python3_FOUND)
        add_test(
            NAME exhibitflow_bytetrack_converter_cleanup
            COMMAND ${CMAKE_COMMAND} -E rm -f
                ${CMAKE_BINARY_DIR}/converted_tracks.jsonl
                ${CMAKE_BINARY_DIR}/converted_tracks.jsonl.tmp
        )

        set_tests_properties(exhibitflow_bytetrack_converter_cleanup PROPERTIES
            FIXTURES_SETUP bytetrack_conversion_clean
        )

        add_test(
            NAME exhibitflow_bytetrack_result_converter
            COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/tools/convert_bytetrack_results.py
                --mot-txt ${PROJECT_SOURCE_DIR}/examples/bytetrack_mot_results.txt
                --output ${CMAKE_BINARY_DIR}/converted_tracks.jsonl
                --sequence-id fixture_01
                --camera-id cam_01
                --fps 25
                --width 384
                --height 288
                --frame-count 3
                --input-frame-base 0
        )

        set_tests_properties(exhibitflow_bytetrack_result_converter PROPERTIES
            FIXTURES_REQUIRED bytetrack_conversion_clean
            FIXTURES_SETUP bytetrack_conversion
        )

        add_test(
            NAME exhibitflow_bytetrack_converter_expected_output
            COMMAND ${CMAKE_COMMAND} -E compare_files
                ${CMAKE_BINARY_DIR}/converted_tracks.jsonl
                ${PROJECT_SOURCE_DIR}/examples/expected_converted_tracks.jsonl
        )

        set_tests_properties(exhibitflow_bytetrack_converter_expected_output PROPERTIES
            FIXTURES_REQUIRED bytetrack_conversion
        )
    endif()
endif()
