# **Rules for Unit Tests:**  

All tests should follow these guidelines:

- **Isolate Tests**: Each unit test MUST BE independent. Avoid shared state so that tests can run in any order.

- **Fast Execution**: Keep tests fast to encourage running them frequently, especially during development.

- **Clear Naming**: Use descriptive names that explain what the test does and what behavior is expected. For example, "shouldReturnErrorWhenInputIsNull".

- **Single Responsibility**: Each test should verify one behavior or aspect of functionality. This improves clarity and maintainability.

- **Setup & Teardown**: Use proper setup and teardown methods to prepare the environment and clean up after tests, ensuring a consistent state.

- **Use Assertive Assertions**: Rely on clear assertions that immediately indicate failure conditions. Consider using multiple assertions only if they don't mask errors.

- **Mock External Dependencies**: Use test doubles (mocks, stubs, fakes) to isolate the unit under test and avoid relying on external systems.

- **Test Edge Cases**: Include tests for typical cases, edge conditions, and failure modes to ensure robust behavior.

- **Maintain Test Readability**: Structure tests so that they're easy to read and understand. Use helper methods to abstract complex preparations.

- **Automate Test Runs**: Integrate tests into continuous integration (CI) to catch errors as early as possible.

- **Document Test Intent**: Include comments or documentation if the test covers non-obvious behavior or complex scenarios.

- **Follow the AAA Pattern (Arrange, Act, Assert)**: Structure tests into clear sections for preparing data, executing the functionality, and asserting the outcomes.

- **Test for Exceptions**: Include tests for expected exceptions to ensure proper error handling.

- **Test for Correctness**: Ensure that the tested code behaves correctly under different conditions and inputs.

---
