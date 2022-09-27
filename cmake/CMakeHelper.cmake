# Macro to add source files to COLMAP library.
macro(LINE_RELATIVE_POSE_ADD_SOURCES)
    set(SOURCE_FILES "")
    foreach(SOURCE_FILE ${ARGN})
        if(SOURCE_FILE MATCHES "^/.*")
            list(APPEND SOURCE_FILES ${SOURCE_FILE})
        else()
            list(APPEND SOURCE_FILES
                 "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
        endif()
    endforeach()
    set(LINE_RELATIVE_POSE_SOURCES ${LINE_RELATIVE_POSE_SOURCES} ${SOURCE_FILES} PARENT_SCOPE)
endmacro(LINE_RELATIVE_POSE_ADD_SOURCES)

