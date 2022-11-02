from loguru import logger

# isort: off
from hvac_fdd_dataset.example import ExampleClass
# isort: on


class TestExample:
    """
    Example unit test class, following pytest conventions (https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html#test-discovery).

    To make pytest actually find and run your tests you need to:
    1. prefix your class name with `Test` (Eg. `class TestExample`)
    2. prefix all your test functions with `test_` (eg `def test_example_function()`)

    To quote the pytest doc: `pytest discovers all tests following its [Conventions for Python test discovery](https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html#test-discovery)),
    so it finds both test_ prefixed functions. There is no need to subclass anything, but make sure to prefix your class with Test otherwise the class will be skipped.`
    """
    test_place: str = None

    def setupClass(self):
        """
        This is run only once before any of the test functions in this class.
        Eg. initialise any variables or resources used by all test functions.
        """
        self.test_place = 'a Unit Test'
        logger.info('Starting test class')

    def teardownClass(self):
        """
        This is run only once after all of the test functions in this class are complete.
        """
        logger.info('Finished test class')

    def test_example_function(self):
        """ tests that the function returns a string """
        ex = ExampleClass(self.test_place)
        result = ex.print_hi_from_place('A Test')
        assert isinstance(
            result, str), 'print_hi_from_place() should only return a string'
        logger.info('Finished running test_example_function()')


if __name__ == '__main__':
    """ Run the test manually - useful for testing. Normally run with `pytest` """
    TestExample().test_example_function()
