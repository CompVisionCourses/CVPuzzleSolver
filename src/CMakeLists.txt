add_executable(CVPuzzleSolver
        main.cpp
        sides_comparison_utils.cpp
)
target_link_libraries(CVPuzzleSolver PRIVATE libbase libimages)

set_target_properties(CVPuzzleSolver PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)