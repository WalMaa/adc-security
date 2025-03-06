class TestingAndValidation:
    def __init__(self, test_cases):
        self.test_cases = test_cases
        self.failed_tests = 0

    def run_tests(self):
        """Executes all test cases and logs failures."""
        for test in self.test_cases:
            print(f"Running: {test['description']}")
            result = test["function"](*test["input"])
            if result == test["expected"]:
                print(f"✅ Passed: {test['description']}")
            else:
                print(f"❌ Failed: {test['description']} - Expected: {test['expected']}, Got: {result}")
                self.failed_tests += 1  # Count failed tests

    def validate_results(self):
        """Validates test results and returns True if all tests pass, False otherwise."""
        self.failed_tests = 0
        self.run_tests()
        return self.failed_tests == 0
