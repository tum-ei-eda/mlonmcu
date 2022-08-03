MACRO(AddAllSubdirs)
    UNSET(LOCAL_SUBDIRS)
    FILE(
        GLOB LOCAL_SUBDIRS
        RELATIVE ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/*
    )

    FOREACH(subdir ${LOCAL_SUBDIRS})
        IF(IS_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${subdir})
            IF(EXISTS ${CMAKE_CURRENT_LIST_DIR}/${subdir}/CMakeLists.txt)
                ADD_SUBDIRECTORY(${CMAKE_CURRENT_LIST_DIR}/${subdir})
                MESSAGE(STATUS "Including sub project ${subdir}.")
            ELSE()
                MESSAGE(
                    WARNING "Directory ${subdir} is not added to the current build because it lacks a CMakeLists.txt"
                )
            ENDIF()
        ENDIF()
    ENDFOREACH()
ENDMACRO()
