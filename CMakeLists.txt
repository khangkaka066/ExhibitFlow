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
    src/core/dma_features.cpp
    src/core/json.cpp
    src/core/track_writer.cpp
    src/detectors/yolox_postprocess.cpp
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

option(EXHIBITFLOW_BUILD_DETECTOR "Build the OpenCV detector CLI when dependencies are available" ON)
set(EXHIBITFLOW_ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of an ONNX Runtime C++ installation")
set(EXHIBITFLOW_TENSORRT_ROOT "" CACHE PATH "Root directory of an NVIDIA TensorRT installation")

if(EXHIBITFLOW_BUILD_DETECTOR)
    find_package(OpenCV QUIET COMPONENTS core imgproc videoio)
    if(NOT OpenCV_FOUND)
        message(STATUS "C++ detector disabled: OpenCV (core, imgproc, videoio) was not found")
    else()
        add_library(exhibitflow_detector_runtime
            src/detectors/yolox_preprocess.cpp
        )

        target_include_directories(exhibitflow_detector_runtime PUBLIC ${PROJECT_SOURCE_DIR}/include)
        target_link_libraries(exhibitflow_detector_runtime PUBLIC exhibitflow_core ${OpenCV_LIBS})
        target_compile_options(exhibitflow_detector_runtime PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
        )

        set(EXHIBITFLOW_DETECTOR_BACKEND_FOUND FALSE)
        find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            HINTS ${EXHIBITFLOW_ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
            PATH_SUFFIXES include include/onnxruntime/core/session
        )
        find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
            HINTS ${EXHIBITFLOW_ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
            PATH_SUFFIXES lib lib64
        )
        if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
            target_sources(exhibitflow_detector_runtime PRIVATE src/detectors/onnxruntime_yolox_detector.cpp)
            target_include_directories(exhibitflow_detector_runtime PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
            target_link_libraries(exhibitflow_detector_runtime PRIVATE ${ONNXRUNTIME_LIBRARY})
            target_compile_definitions(exhibitflow_detector_runtime PUBLIC EXHIBITFLOW_WITH_ONNXRUNTIME=1)
            set(EXHIBITFLOW_DETECTOR_BACKEND_FOUND TRUE)
        else()
            message(STATUS "ONNX Runtime backend disabled: set EXHIBITFLOW_ONNXRUNTIME_ROOT or ONNXRUNTIME_ROOT")
        endif()

        find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS ${EXHIBITFLOW_TENSORRT_ROOT} ENV TENSORRT_ROOT
            PATH_SUFFIXES include
        )
        find_library(TENSORRT_NVINFER_LIBRARY NAMES nvinfer
            HINTS ${EXHIBITFLOW_TENSORRT_ROOT} ENV TENSORRT_ROOT
            PATH_SUFFIXES lib lib64
        )
        find_path(CUDA_RUNTIME_INCLUDE_DIR cuda_runtime_api.h
            HINTS ENV CUDA_PATH ENV CUDA_HOME
            PATH_SUFFIXES include
        )
        find_library(CUDA_RUNTIME_LIBRARY NAMES cudart
            HINTS ENV CUDA_PATH ENV CUDA_HOME
            PATH_SUFFIXES lib lib64 lib/x64
        )
        if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIBRARY AND CUDA_RUNTIME_INCLUDE_DIR AND CUDA_RUNTIME_LIBRARY)
            target_sources(exhibitflow_detector_runtime PRIVATE src/detectors/tensorrt_yolox_detector.cpp)
            target_include_directories(exhibitflow_detector_runtime PRIVATE
                ${TENSORRT_INCLUDE_DIR}
                ${CUDA_RUNTIME_INCLUDE_DIR}
            )
            target_link_libraries(exhibitflow_detector_runtime PRIVATE
                ${TENSORRT_NVINFER_LIBRARY}
                ${CUDA_RUNTIME_LIBRARY}
            )
            target_compile_definitions(exhibitflow_detector_runtime PUBLIC EXHIBITFLOW_WITH_TENSORRT=1)
            set(EXHIBITFLOW_DETECTOR_BACKEND_FOUND TRUE)
        else()
            message(STATUS "TensorRT backend disabled: set EXHIBITFLOW_TENSORRT_ROOT/TENSORRT_ROOT and CUDA_PATH")
        endif()

        if(EXHIBITFLOW_DETECTOR_BACKEND_FOUND)
            add_executable(exhibitflow_detector apps/detector_main.cpp)
            target_link_libraries(exhibitflow_detector PRIVATE exhibitflow_detector_runtime)
            target_compile_options(exhibitflow_detector PRIVATE
                $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
            )
            set_target_properties(exhibitflow_detector PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            )
        else()
            message(STATUS "C++ detector executable disabled: no inference backend was found")
        endif()
    endif()
endif()

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

    add_executable(exhibitflow_dma_feature_tests
        tests/test_dma_features.cpp
    )

    target_link_libraries(exhibitflow_dma_feature_tests PRIVATE exhibitflow_core)

    target_compile_options(exhibitflow_dma_feature_tests PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    )

    add_test(NAME exhibitflow_dma_feature_tests COMMAND exhibitflow_dma_feature_tests)

    add_executable(exhibitflow_yolox_postprocess_tests
        tests/test_yolox_postprocess.cpp
    )

    target_link_libraries(exhibitflow_yolox_postprocess_tests PRIVATE exhibitflow_core)

    target_compile_options(exhibitflow_yolox_postprocess_tests PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
    )

    add_test(NAME exhibitflow_yolox_postprocess_tests COMMAND exhibitflow_yolox_postprocess_tests)

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
            --device cpu
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
