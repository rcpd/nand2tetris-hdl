""""
A rough example of a Python implementation of the Nand2Tetris HDL
"""


class NandGate(object):
    """
    NAND is a primitive implemented at the hardware level so need to the logic ourselves
    All subsequent gates can be expressed via increasingly complex abstractions of NAND
    This is typically provided to the student as a built-in class for Nand2Tetris
    All subsequent gates must be implemented by the student
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.out = self.evaluate(a, b)

    def evaluate(self, a, b):
        if a and b:
            return False
        else:
            return True


class NotGate(object):
    """
    For a single input, return the opposite
    """
    def __init__(self, a):
        self.a = a
        self.out = self.evaluate(a)

    def evaluate(self, a):
        """
        A single input/output can still have a one-to-many relationship with other gates
        In this case it is passed to both inputs of the NAND gate
        """
        return NandGate(a, a).out


class AndGate(object):
    """
    This implementation is left to the student in Nand2Tetris
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.out = self.evaluate(a, b)

    def evaluate(self, a, b):
        """
        Now we can combine a NAND with a NOT to get a regular AND
        """
        nand_out = NandGate(a, b).out
        return NotGate(nand_out).out


class OrGate(object):
    def __init__(self, a, b):
        raise NotImplementedError


class NorGate(object):
    def __init__(self, a, b):
        raise NotImplementedError


class XorGate(object):
    def __init__(self, a, b):
        raise NotImplementedError


class XNorGate(object):
    def __init__(self, a, b):
        raise NotImplementedError


def main():
    """
    Sanity check out truth tables for each gate as implemented
    These unit tests are typically provided to the student in Nand2Tetris so they can confirm their results
    """
    assert NandGate(True, True).out is False
    assert NandGate(True, False).out is True
    assert NandGate(False, False).out is True

    assert NotGate(True).out is False
    assert NotGate(False).out is True

    assert AndGate(True, True).out is True
    assert AndGate(True, False).out is False
    assert AndGate(False, False).out is False


if __name__ == "__main__":
    main()
