execute_process(
  COMMAND "${TEST_EXECUTABLE}" "${TEST_OPTION}" "${TEST_VALUE}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)

if(NOT result MATCHES "^-?[0-9]+$")
  message(FATAL_ERROR "command did not run: ${result}")
endif()
if(result EQUAL 0)
  message(FATAL_ERROR "command unexpectedly succeeded")
endif()

set(output "${stdout}${stderr}")
string(FIND "${output}" "${EXPECTED_MESSAGE}" message_position)
if(message_position EQUAL -1)
  message(FATAL_ERROR
    "expected output not found: ${EXPECTED_MESSAGE}\nActual output:\n${output}")
endif()
