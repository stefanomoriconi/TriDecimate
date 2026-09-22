# Unit tests for the decimate library.
#
# The heavy correctness logic is exercised through the Python/ctypes layer (see
# test_decimate.py), which is language-agnostic and validates the shared
# library ABI across platforms. Here we add a small native C smoke test that
# links directly against the library and exercises the public C ABI, plus
# wiring so `ctest` also drives the Python test-suite when a Python interpreter
# is available.

add_executable(decimate_smoke_test decimate_smoke_test.c)
target_link_libraries(decimate_smoke_test PRIVATE decimate::decimate)

add_test(NAME smoke_test COMMAND decimate_smoke_test)

# ---- Optional Python integration tests -------------------------------------
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/python/decimate.py")
    # The Python test-suite discovers and loads the freshly built library via
    # the DECIMATE_LIB env var set by the test harness below.
    add_test(
        NAME python_tests
        COMMAND ${Python3_EXECUTABLE} -m pytest -q "${CMAKE_SOURCE_DIR}/tests/test_decimate.py"
    )
    # Make the built library path available to the wrapper.
    set(_lib_dir "$<TARGET_FILE_DIR:decimate>")
    if(WIN32)
        set_tests_properties(python_tests PROPERTIES ENVIRONMENT "DECIMATE_LIB_DIR=${_lib_dir};$ENV{PATH}")
    else()
        set_tests_properties(python_tests PROPERTIES ENVIRONMENT "DECIMATE_LIB_DIR=${_lib_dir}")
    endif()
endif()
